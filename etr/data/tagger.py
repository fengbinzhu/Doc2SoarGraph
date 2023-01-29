import re
from etr.data.tag_constant import *
import numpy as np


class Tagger:

    @staticmethod
    def _tag(text, pattern, tag_name=None, tag_code=None):
        result = []
        tags = [0] * len(text)

        for match in re.finditer(pattern, text):
            result.append((match.group(), list(match.span())))
            match_start = match.start()
            match_end = match.end()
            tags[match_start:match_end] = [tag_code] * len(match.group())

        # while re.search(pattern, text):
        #     match = re.search(pattern, text)
        #     result.append((match.group(), list(match.span())))
        #     match_start = match.start()
        #     match_end = match.end()
        #     tags[match_start:match_end] = [tag_code] * len(match.group())
        #     text = text[:match_start] + 'µ' * len(match.group()) + text[match_end:]
        return result, tags

    @staticmethod
    def _match_content(text, tag=None):
        tag_name, pattern, _ = tag
        while re.search(pattern, text):
            match = re.search(pattern, text)
            text = text[:match.start()] + 'µ' * len(match.group()) + text[match.end():]
        return text

    @staticmethod
    def _mask_connected_amount(text, match):
        """
        it is hard to check if a space is part of amount or a separator for two amounts
        . This method is designed to handle "2,191,000,000.00 2,746,707,828.64".
        """
        long_amount = match.end() - match.start() > 20
        single_space = len(re.findall(' ', match.group())) == 1
        two_decimals = len(re.findall(r'(?:\.|,)\d{2}(?:$| )', match.group())) == 2
        if long_amount and single_space and two_decimals:
            space_idx = match.start() + match.group().index(' ')
            text = text[:space_idx] + 'µ' + text[space_idx + 1:]
        return text

    @staticmethod
    def _extract(text, tag=None, exclude_tags=[]):
        tag_name, pattern, tag_code = tag
        if tag_name == '<NUMBER>':
            for it in exclude_tags:
                text = Tagger._match_content(text, tag=it)
        return Tagger._tag(text, pattern, tag_name, tag_code)

    @staticmethod
    def convert(text, tags=[TAG_NUMBER, TAG_DATE]):
        text_tag_offsets = {}
        text_tag_codes = np.array([0] * len(text))
        for tag in tags:
            tag_name = tag[0]
            exclude_tags = []
            if tag == TAG_NUMBER and len(tags) > 1:
                exclude_tags = [TAG_DATE]
            tag_offsets, tags = Tagger._extract(text, tag, exclude_tags)
            text_tag_codes += np.array(tags)
            text_tag_offsets[tag_name] = tag_offsets
        return text_tag_offsets, list(text_tag_codes)
