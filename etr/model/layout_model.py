from transformers import LayoutLMv2Processor, LayoutLMv2Model, LayoutLMv2FeatureExtractor
from transformers import LayoutLMTokenizer, LayoutLMModel
from PIL import Image
import torch
import numpy as np

feature_extractor = LayoutLMv2FeatureExtractor(apply_ocr=False)

def normalize_bbox(bbox, width, height):
    return [
        int(1000 * (bbox[0]/ width)),
        int(1000 * (bbox[1]/ height)),
        int(1000 * (bbox[2]/ width)),
        int(1000 * (bbox[3] / height))
    ]
def extract_feature_from_image(image:Image):
    encoding = feature_extractor(image, return_tensors="np")
    return encoding['pixel_values']


def layoutlm_process_test(words, bboxes):
    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    model = LayoutLMModel.from_pretrained("microsoft/layoutlm-base-uncased")
    token_boxes = []
    for word, box in zip(words, bboxes):
        word_tokens = tokenizer.tokenize(word)
        token_boxes.extend([box] * len(word_tokens))
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]
    encoding = tokenizer(" ".join(words), return_tensors="pt")
    for name, tensor in encoding.items():
        print(f'{name}: {tensor.shape}')
    bbox = torch.tensor([token_boxes])

    outputs = model( input_ids=encoding['input_ids'],
                     bbox=bbox,
                     attention_mask=encoding['attention_mask'],
                     token_type_ids=encoding['token_type_ids'])
    print(outputs)



def layoutlm_v2_process_test(words:list, bboxes:list, image:Image):

    processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
    model = LayoutLMv2Model.from_pretrained('microsoft/layoutlmv2-base-uncased')
    encoding = processor(image, words, boxes=bboxes, return_tensors="pt")
    for name, tensor in encoding.items():
        print(f'{name}: {tensor.shape}')

    encoding.pop('image')
    for name, tensor in encoding.items():
        print(f'{name}: {tensor.shape}')

    outputs = model(**encoding)
    print(outputs[0].shape)
    print(outputs[0][:,:4,:].shape)
    # print(outputs)
    
def layoutlm_v2_fe_test(words:list, bboxes:list, image:Image):
    pixels = extract_feature_from_image(image)
    print(pixels)
    print(np.array(pixels).shape)
    
if __name__ == '__main__':

    words = ["hello", "world"]
    boxes = [[1, 2, 3, 4], [5, 6, 7, 8]]  # make sure to normalize your bounding boxes
    image = Image.open('./dataset_tatdqa/tat_docs/test/0ba33bc0610e557810de948f4248719e_1.png').convert("RGB")
    # layoutlm_v2_fe_test(words, boxes, image)
    layoutlm_v2_process_test(words, boxes, image)
    # layoutlm_process(words, boxes)