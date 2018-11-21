import traceback
from argparse import Namespace

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data
from ocr.net.str import Model
from ocr.utils import (AttnLabelConverter, Averager, CTCLabelConverter,
                       normalize)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class STROCR():
    def __init__(self,
                 weights_path,
                 chars_set_path,
                 input_height=32,
                 input_width=100,
                 batch_max_length=25,
                 padding=True,
                 rgb=False,
                 transformation='TPS',
                 feature_extraction='ResNet',
                 sequence_modeling='BiLSTM',
                 prediction='Attn',
                 num_fiducial=20,
                 output_channel=512,
                 hidden_size=256,
                 debug=False):
        """ STR OCR model.

        Parameters
        ----------
            weights_path : str
                Path to trained model
            text_threshold : float
                Text confidence threshold (default is 0.7)
            low_text : float
                Text low-bound score (default is 0.4)
            link_threshold : float
                Link confidence threshold (default is 0.4)
            canvas_size : int
                Image size for inference (default is 1280)
            mag_ratio : int
                Image magnification ratio (default is 1.5)
            debug : bool
                Enable debug information to be generated at './debugs/' (default is False)

        """
        # Create config
        self.debug = debug
        self.chars_set = open(chars_set_path, 'r').read().replace('\n', '')
        if prediction == 'CTC':
            self.converter = CTCLabelConverter(self.chars_set)
        elif prediction == 'Attn':
            self.converter = AttnLabelConverter(self.chars_set)
        else:
            raise ValueError(
                'Unknown label converter: {}. Only support CTC and Attn'.format(prediction))
        num_class = len(self.converter.character)
        input_channel = 3 if rgb else 1
        config = {
            'Transformation': transformation,
            'FeatureExtraction': feature_extraction,
            'SequenceModeling': sequence_modeling,
            'Prediction': prediction,
            'num_fiducial': num_fiducial,
            'imgH': input_height,
            'imgW': input_width,
            'input_channel': input_channel,
            'output_channel': output_channel,
            'hidden_size': hidden_size,
            'num_class': num_class,
            'batch_max_length': batch_max_length,
        }
        self.config = Namespace(**config)

        # Create model
        model = Model(self.config)
        self.model = torch.nn.DataParallel(model).to(device)

        # Load weights
        print('Loading pretrained model from', weights_path)
        self.model.load_state_dict(torch.load(
            weights_path, map_location=device))

        # Set inference mode
        self.model.eval()
        cudnn.benchmark = True
        cudnn.deterministic = True

    def predict(self, imgs: [], batch_size=1):
        imgs = [normalize(x) for x in imgs]
        results = []
        confidence_scores = []
        for i in range(0, len(imgs), batch_size):
            imgs_batch = imgs[i:i+batch_size]
            imgs_batch = self.batch_img(imgs_batch)
            results_batch, confidence_scores_batch = self._predict(imgs_batch)
            results += results_batch
            confidence_scores += confidence_scores_batch
        return results, confidence_scores

    def batch_img(self, data):
        """Pad and batch a sequence of images."""
        c = data[0].size(0)
        h = max([t.size(1) for t in data])
        w = max([t.size(2) for t in data])
        imgs = torch.zeros(len(data), c, h, w).fill_(1)
        for i, img in enumerate(data):
            imgs[i, :, 0:img.size(1), 0:img.size(2)] = img
        return imgs

    def _predict(self, image_tensors):
        try:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor(
                [self.config.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(
                batch_size, self.config.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in self.config.Prediction:
                preds = self.model(image, text_for_pred).log_softmax(2)
                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                preds_index = preds_index.view(-1)
                preds_str = self.converter.decode(
                    preds_index.data, preds_size.data)
            else:
                preds = self.model(image, text_for_pred, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode(preds_index, length_for_pred)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)

            results = []
            confidence_scores = []
            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                if 'Attn' in self.config.Prediction:
                    pred_EOS = pred.find('[s]')
                    # prune after "end of sentence" token ([s])
                    pred = pred[:pred_EOS]
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                results.append(pred)
                confidence_scores.append(
                    float(confidence_score.detach().numpy()))
        except Exception as ex:
            print(traceback.format_exc())
            return [""], [0]
        return results, confidence_scores