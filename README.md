# Raft Extractor API

## Objectives
1. Detect Regions of text in Real world images
    - Draw bounding boxes around all regions of text found
    - Example Image: ![](https://i.stack.imgur.com/dqs8L.jpg)
2. (based on 1) - Extract (OCR) text out of the bounding boxes found in Real world images in step 1
    - Straighten the text if it is skewed

## Goal
A Web API that:
1. Takes an image as input and returns X and Y coordinates for bounding boxes (step 1 in objectives)
2. Takes an image as input and returns the text extracted from the image as string

