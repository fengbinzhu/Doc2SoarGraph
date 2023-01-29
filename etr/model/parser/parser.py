import re

def create_date_pattern():
    # %Y.%m.%d
    pattern1 = r'(?<!\d|\-|\/|\.|\*)(19|20)?\d\d[.](0[1-9]|1[0-2])[.](0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)'
    # %Y%m%d
    pattern1_2 = r'(?<!\d|\-|\/|\.|\*)(19|20)?\d\d(0[1-9]|1[0-2])(0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)'
    # %Y-%m-%d
    pattern2 = r'(?<!\d|\-|\/|\.|\*)(19|20)?\d\d[-](0[1-9]|1[0-2])[-](0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)'
    # %Y/%m/%d
    pattern3 = r'(?<!\d|\-|\/|\.|\*)(19|20)?\d\d[\/](0[1-9]|1[0-2])[\/](0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)'
    # %m.%d.%Y
    pattern4 = r'(?<!\d|\-|\/|\.|\#|\*)(0[1-9]|1[0-2])[.](0[1-9]|[1-2][0-9]|3[0-1])([.](19|20)?\d\d)?(?!\d|\-|\/|\.)'
    # %m-%d-%Y
    pattern5 = r'(?<!\d|\-|\/|\.|\#|\*)(0[1-9]|1[0-2])[-](0[1-9]|[1-2][0-9]|3[0-1])([-](19|20)?\d\d)?(?!\d|\-|\/|\.)'
    # %m/%d/%Y
    pattern6 = r'(?<!\d|\-|\/|\.|\#|\*)(0[1-9]|1[0-2])[\/](0[1-9]|[1-2][0-9]|3[0-1])([\/](19|20)?\d\d)?(?!\d|\-|\/|\.)'
    # %d.%m.%Y
    pattern7 = r'(?<!\d|\-|\/|\.|\#|\*)(0[1-9]|[1-2][0-9]|3[0-1])[.](0?[1-9]|1[0-2])([.](19|20)?\d\d)?(?!\d|\-|\/|\.)'
    # %d-%m-%Y
    pattern8 = r'(?<!\d|\-|\/|\.|\#|\*)(0[1-9]|[1-2][0-9]|3[0-1])[-](0?[1-9]|1[0-2])([-](19|20)?\d\d)?(?!\d|\-|\/|\.)'
    # %d/%m/%Y
    pattern9 = r'(?<!\d|\-|\/|\.|\#|\*)(0[1-9]|[1-2][0-9]|3[0-1])[\/](0?[1-9]|1[0-2])([\/](19|20)?\d\d)?(?!\d|\-|\/|\.)'
    # %Y[./\- ,](English words)[./\- ,](English ordinal or number only)
    pattern10 = r'(?<!\d|\-|\/|\.|\*)(19|20)?\d\d\s*[\.\/ \-\,]?\s*((jan|Jan|JAN)(uary|UARY)?|(feb|Feb|FEB)' \
                r'(ruary|RUARY)?|(mar|Mar|MAR)(ch|CH)?|(apr|Apr|APR)(il|IL)?|(may|May|MAY)|(jun(e|)?|Jun(e|)?' \
                r'|JUN(E)?)|(jul(y)?|Jul(y)?|JUL(Y)?)|(aug|Aug|AUG)(ust|UST)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember' \
                r'|TEMBER)?|(oct|Oct|OCT)(ober|OBER)?|(nov|Nov|NOV)(ember|EMBER)?|(dec|Dec|DEC)(ember|EMBER)?)' \
                r'\s*[\.\/ \-\,]?\s*(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?(?!\d|\-|\/|[a-zA-Z])'
    # (English words)[./\- ,](English ordinal or number only)[./\- ,]%Y
    pattern11 = r'(?<!\d|\-|\/|\.|\*)((jan|Jan|JAN)(uary|UARY)?|(feb|Feb|FEB)(ruary|RUARY)?|(mar|Mar|MAR)' \
                r'(ch|CH)?|(apr|Apr|APR)(il|IL)?|(may|May|MAY)|(jun(e|)?|Jun(e|)?|JUN(E)?)|(jul(y)?|Jul(y)?' \
                r'|JUL(Y)?)|(aug|Aug|AUG)(ust|UST)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember|TEMBER)?|(oct|Oct|OCT)' \
                r'(ober|OBER)?|(nov|Nov|NOV)(ember|EMBER)?|(dec|Dec|DEC)(ember|EMBER)?)\s*[\.\/ \-\,]?' \
                r'\s*(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?(\s*[\.\/ \-\,]?\s*(19|20)\d\d)?(?!\d|\-|\/|[a-z|A-Z])'
    # (English ordinal or number only)[./\- ,](English words)[./\- ,]%Y
    pattern12 = r'(?<!\d|\-|\/|\.|\*)(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?\s*[\.\/ \-\,]?\s*' \
                r'((jan|Jan|JAN)(uary|UARY)?|(feb|Feb|FEB)(ruary|RUARY)?|(mar|Mar|MAR)(ch|CH)?|' \
                r'(apr|Apr|APR)(il|IL)?|(may|May|MAY)|(jun(e|)?|Jun(e|)?|JUN(E)?)|(jul(y)?|Jul(y)?|JUL(Y)?)' \
                r'|(aug|Aug|AUG)(ust|UST)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember|TEMBER)?|(oct|Oct|OCT)(ober|OBER)?|' \
                r'(nov|Nov|NOV)(ember|EMBER)?|(dec|Dec|DEC)(ember|EMBER)?)(\s*[\.\/ \-\,]?\s*(19|20)?\d\d)?' \
                r'(?!\d|\-|\/|[a-z|A-Z])'

    date_pattern = '|'.join([
        pattern1,
        pattern1_2,
        pattern2,
        pattern3,
        pattern4,
        pattern5,
        pattern6,
        pattern7,
        pattern8,
        pattern9,
        pattern10,
        pattern11,
        pattern12
    ])
    return date_pattern

DATE_PATTERN = create_date_pattern()

def date_parser(value):
    """
    :param: parsed_value is some value of date type

    output: value after parsing
    """
    test = re.search(DATE_PATTERN, str(value))
    if test:
        return test.group()
    raise ValueError(f'Unable to parse {parsed_value} to date')


def num_parser(value):
    try:
        return float(value)
    except Exception:
        raise ValueError(f'Unable to parse {value} to number')