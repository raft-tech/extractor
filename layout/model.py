# -*- coding: utf-8 -*-
import logging
import os
import time
import uuid
import tempfile
from zipfile import ZipFile

import cv2
import numpy as np
import torch
from layout.utils import craft_utils, imgproc
from torch.autograd import Variable

logging.getLogger(__name__).addHandler(logging.NullHandler())

logger = logging.getLogger(__name__)


class CRAFTLayout:
    def __init__(self,
                 weights_path,
                 link_refine=False,
                 text_threshold=0.7,
                 low_text=0.4,
                 link_threshold=0.4,
                 canvas_size=1280,
                 mag_ratio=1.5,
                 debug=False):
        """ CRAFT Layout model.

        Parameters
        ----------
            weights_path : str
                Path to zipped trained model
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

        self.cuda = torch.cuda.is_available()

        self.link_refine = link_refine
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.debug = debug
        if self.debug:
            os.makedirs('./debugs', exist_ok=True)

        # Unzip weights
        with tempfile.TemporaryDirectory() as temp:
            with ZipFile(weights_path, 'r') as zipObj:
                zipObj.extractall(temp)
            craft_weights_path = os.path.join(temp, 'craft.pth')
            refine_net_weights_path = os.path.join(
                temp, 'refiner.pth')

            # Load CRAFT
            self.net = craft_utils.load_model(
                craft_weights_path, cuda=self.cuda, eval=True)

            # Load LinkRefiner
            if self.link_refine:
                self.refine_net = craft_utils.load_model(
                    refine_net_weights_path, cuda=self.cuda, eval=True, refine_net=True)
            else:
                self.refine_net = None

    def _convert_format(self, result):
        converted = []
        for index, line in enumerate(result):
            converted.append({
                "id": index,
                "location": [tuple([int(each[0]), int(each[1])]) for each in line],
            })
        return converted

    def process(self, image, debug_path=None):
        start_time = time.time()

        # resize
        img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(
            image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
        if self.cuda:
            x = x.cuda()

        # forward pass
        y, feature = self.net(x)

        # make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()

        # refine link
        if self.link_refine:
            y_refiner = self.refine_net(y, feature)
            score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(
            score_text, score_link, self.text_threshold, self.link_threshold, self.low_text, poly=True)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]

        if self.debug:
            logger.info('Took {} secs'.format(time.time() - start_time))
            render_img = score_text.copy()
            render_img = np.hstack((render_img, score_link))
            ret_score_text = imgproc.cvt2HeatmapImg(render_img)
            debug_path = os.path.join(
                './debugs/', str(uuid.uuid4())) if debug_path is None else debug_path
            os.makedirs(debug_path, exist_ok=True)
            cv2.imwrite(os.path.join(
                debug_path, 'mask_layout.jpg'), ret_score_text)
            craft_utils.saveResult(os.path.join(debug_path, 'layout.jpg'), image[:, :, ::-1], polys, dirname=debug_path)

        return self._convert_format(polys)