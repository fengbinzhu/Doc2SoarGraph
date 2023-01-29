
from etr.data.tagger import Tagger
from etr.data.tag_constant import TAG_DATE, TAG_NUMBER, TAG_AMOUNT
import re


def doc_tagger_test():
    text = 'for the years ended december 31 , 2010 , 2009 , and 2008 , the potential anti-dilutive share conversions were 256868 shares , 1230881 shares , and 638401 shares , respectively '
    text = 'related party transactions the ace foundation 2013 bermuda is an unconsolidated not-for-profit organization whose primary purpose is to fund charitable causes in bermuda '
    text = 'bermuda subsidiaries 2010'
    text = '$ 2430'
    text = '1,121.48 1.00 i am 2,430,120  -234.0, 34.3%, 1.20  december 31 , 2010, based on a 365 day year and the average 321.00  321,000.12 sales price per barrel listed above , 20061 what was the total refined product sales revenue for 2006?'
    # text = 'management 2019s financial discussion and analysis 2010 compared to 2009 net revenue consists of operating revenues net of : 1 ) fuel , fuel-related expenses , and gas purchased for resale , 2 ) purchased power expenses , and 3 ) other regulatory charges ( credits )'
    # text = 'in 2010 what was the net change in net revenue in millions'
    # text = 'the decrease was partially offset by an increase of 120021 -3.90 million in legal 23.4% expenses due to the deferral in 2010 of certain litigation expenses in accordance with regulatory treatment '
    # text_tag_offsets, text_tag_codes = Tagger.convert(text)
    # print(text_tag_offsets)
    # print(text_tag_codes)

    print('For\u00a0the\u00a0years\u00a0ended\u00a0December\u00a031,\u00a02019\u00a0and\u00a0December\u00a031,\u00a02018,\u00a0the\u00a0Company\u00a0had\u00a0a\u00a0net\u00a0loss\u00a0available\u00a0to\u00a0common\u00a0shareholders\u00a0and,\u00a0as\u00a0a\u00a0result,\u00a0all\u00a0common\u00a0stock equivalents\u00a0were\u00a0excluded\u00a0from\u00a0the\u00a0computation\u00a0of\u00a0diluted\u00a0EPS\u00a0as\u00a0their\u00a0inclusion\u00a0would\u00a0have\u00a0been\u00a0anti-dilutive.\u00a0\u00a0For\u00a0the\u00a0year\u00a0ended\u00a0December\u00a031,\u00a02017,\u00a0awards under\u00a0the\u00a0Company\u2019s\u00a0stock-based\u00a0compensation\u00a0plans\u00a0for\u00a0common\u00a0shares\u00a0of\u00a00.2\u00a0million,\u00a0were\u00a0excluded\u00a0from\u00a0the\u00a0computation\u00a0of\u00a0diluted\u00a0EPS\u00a0as\u00a0their\u00a0inclusion would\u00a0have\u00a0been\u00a0anti-dilutive.\u00a0\u00a0For\u00a0all\u00a0periods\u00a0presented,\u00a0preferred\u00a0stock\u00a0convertible\u00a0into\u00a00.9\u00a0million\u00a0common\u00a0shares\u00a0was\u00a0excluded\u00a0as\u00a0it\u00a0was\u00a0anti-dilutive.')
    # print(list(re.finditer('(\\-|\\+)?\\d+[,\\d]+(\\.\\d+)?%?|(\\-|\\+)?\\d+(\\.\\d+)?%?|\\d+%?', text)))
    # assert len(text) == len(tags)
    # amounts, tags = Tagger._extract(text, TAG_AMOUNT)
    # print(amounts)
    # print(tags)
    # assert len(text) == len(tags)
    # nums,  tags = Tagger._extract(text, TAG_NUMBER)
    # print(nums)
    # print(tags)
    # assert len(text) == len(tags)


doc_tagger_test()