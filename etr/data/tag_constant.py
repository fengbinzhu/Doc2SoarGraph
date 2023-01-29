
FULL_YEAR = '(?<!\d|\-|\/|\.|\*)(20)\d{2}(?!\d|\-|\/)'
YEAR = '(19|20)?\d\d'
MONTH_1 = '(0[1-9]|1[0-2])'
MONTH_2 = '((jan|Jan|JAN)(uary|UARY)?|(feb|Feb|FEB)(ruary|RUARY)?|(mar|Mar|MAR)(ch|CH)?|(apr|Apr|APR)(il|IL)?|(may|May|MAY)|(jun(e|)?|Jun(e|)?|JUN(E)?)|(jul(y)?|Jul(y)?|JUL(Y)?)|(aug|Aug|AUG)(ust|UST)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember|TEMBER)?|(oct|Oct|OCT)(ober|OBER)?|(nov|Nov|NOV)(ember|EMBER)?|(dec|Dec|DEC)(ember|EMBER)?)'
DAY_1 = '(0[1-9]|[1-2][0-9]|3[0-1])'
DAY_2 = '(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?'
CONNECTOR_1 = '[{}]'
CONNECTOR_2 = '\s*[\.\/ \-\,]?\s*'

# yyyy[-./ ]mm[-./ ]dd or yy[-./ ]mm[-./ ]dd
DATE_1 = '(?<!\d|\-|\/|\.|\*)'+YEAR+CONNECTOR_1+MONTH_1+CONNECTOR_1+DAY_1+'(?!\d|\-|\/|\.)'

# mm[-./ ]dd[-./ ]yyyy or mm[-./ ]dd[-./ ]yy or mm[-./ ]dd
DATE_2 = '(?<!\d|\-|\/|\.|\#|\*)'+MONTH_1+CONNECTOR_1+DAY_1+'('+CONNECTOR_1+YEAR+')?'+'(?!\d|\-|\/|\.)'

# dd[-./ ]mm[-./ ]yyyy or dd[-./ ]mm[-./ ]yy or dd[-./ ]mm
DATE_3 = '(?<!\d|\-|\/|\.|\#|\*)'+DAY_1+CONNECTOR_1+MONTH_1+'('+CONNECTOR_1+YEAR+')?'+'(?!\d|\-|\/|\.)'

# yyyyMONTHdd or yyMONTHdd
MONTH_DATE_1 = '(?<!\d|\-|\/|\.|\*)'+YEAR+CONNECTOR_2+MONTH_2+CONNECTOR_2+DAY_2+'(?!\d|\-|\/|[a-zA-Z])'

# MONTHddyyyy or MONTHddyy
MONTH_DATE_2 = '(?<!\d|\-|\/|\.|\*)'+MONTH_2+CONNECTOR_2+DAY_2+CONNECTOR_2+YEAR+'(?!\d|\-|\/|[a-z|A-Z])'

# ddMONTHyyyy or ddMONTHyy
MONTH_DATE_3 = '(?<!\d|\-|\/|\.|\*)'+DAY_2+CONNECTOR_2+MONTH_2+CONNECTOR_2+YEAR+'(?!\d|\-|\/|[a-z|A-Z])'

DATE_PATTERN = ''
SIGN_LIST = ['.', '-', '\/']
DATE_LIST = [DATE_1, DATE_2, DATE_3]
MONTH_DATE_LIST = [MONTH_DATE_1, MONTH_DATE_2, MONTH_DATE_3]
for date in DATE_LIST:
    for sign in SIGN_LIST:
        DATE_PATTERN += '|'+date.format(sign, sign)
for MONTH_DATE in MONTH_DATE_LIST:
    DATE_PATTERN += '|'+MONTH_DATE

DATE_PATTERN += '|'+ FULL_YEAR
DATE_PATTERN = DATE_PATTERN[1:]

#(?<!\d|\-|\/|\.)(19|20)?\d\d[.](0[1-9]|1[0-2])[.](0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.)(19|20)?\d\d[-](0[1-9]|1[0-2])[-](0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.)(19|20)?\d\d[\/](0[1-9]|1[0-2])[\/](0[1-9]|[1-2][0-9]|3[0-1])(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.|\#)(0[1-9]|1[0-2])[.](0[1-9]|[1-2][0-9]|3[0-1])([.](19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.|\#)(0[1-9]|1[0-2])[-](0[1-9]|[1-2][0-9]|3[0-1])([-](19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.|\#)(0[1-9]|1[0-2])[\/](0[1-9]|[1-2][0-9]|3[0-1])([\/](19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.|\#)(0[1-9]|[1-2][0-9]|3[0-1])[.](0[1-9]|1[0-2])([.](19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.|\#)(0[1-9]|[1-2][0-9]|3[0-1])[-](0[1-9]|1[0-2])([-](19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.|\#)(0[1-9]|[1-2][0-9]|3[0-1])[\/](0[1-9]|1[0-2])([\/](19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.)(19|20)?\d\d\s*[\.\/ \-\,]?\s*((jan|Jan|JAN)(uary)?|(feb|Feb|FEB)(ruary)?|(mar|Mar|MAR)(ch)?|(apr|Apr|APR)(il)?|(may|May|MAY)|(jun(e)?|Jun(e)?|JUN(E)?)|(jul(y)?|Jul(y)?|JUL(Y)?)|(aug|Aug|AUG)(ust)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember)?|(oct|Oct|OCT)(ober)?|(nov|Nov|NOV)(ember)?|(dec|Dec|DEC)(ember)?)\s*[\.\/ \-\,]?\s*(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.)((jan|Jan|JAN)(uary)?|(feb|Feb|FEB)(ruary)?|(mar|Mar|MAR)(ch)?|(apr|Apr|APR)(il)?|(may|May|MAY)|(jun(e)?|Jun(e)?|JUN(E)?)|(jul(y)?|Jul(y)?|JUL(Y)?)|(aug|Aug|AUG)(ust)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember)?|(oct|Oct|OCT)(ober)?|(nov|Nov|NOV)(ember)?|(dec|Dec|DEC)(ember)?)\s*[\.\/ \-\,]?\s*(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?(\s*[\.\/ \-\,]?\s*(19|20)?\d\d)?(?!\d|\-|\/|\.)|(?<!\d|\-|\/|\.)(0?[1-9]|[1-2][0-9]|3[0-1])(st|nd|rd|th)?\s*[\.\/ \-\,]?\s*((jan|Jan|JAN)(uary)?|(feb|Feb|FEB)(ruary)?|(mar|Mar|MAR)(ch)?|(apr|Apr|APR)(il)?|(may|May|MAY)|(jun(e)?|Jun(e)?|JUN(E)?)|(jul(y)?|Jul(y)?|JUL(Y)?)|(aug|Aug|AUG)(ust)?|(sep(t)?|Sep(t)?|SEP(T)?)(tember)?|(oct|Oct|OCT)(ober)?|(nov|Nov|NOV)(ember)?|(dec|Dec|DEC)(ember)?)(\s*[\.\/ \-\,]?\s*(19|20)?\d\d)?(?!\d|\-|\/|\.)

# TAG_NUMBER = ('<NUMBER>', "[+-]?\d+(\.\d+)?|[+-]?\.\d+%?", 1)
# TAG_NUMBER = ('<NUMBER>', "(\-|\+)?\d+[,\d]+(\.\d+)?%?|(\-|\+)?\d+(\.\d+)?%?|\d+%?", 1)

TAG_NUMBER = ('<NUMBER>', "\d+(,\d+)+(\.\d+)?|\d+(\.\d+)?|\d+", 1)
TAG_AMOUNT = ('<AMOUNT>', '-?(?<!\.|\,|\d)(\d{1,3}(?: *[,.] *?\d *\d *\d *)*(?:[,.](?<!\d)\d{2})(?!\d))(?!\.|\,)', 1)
TAG_DATE = ('<DATE>', FULL_YEAR, 2) # only use full year 20xx
TAG_POSTCODE = ('<POSTCODE>', "(?<!\d)[0-9]{5,6}(?!\d|\-|\.)")
TAG_CURRENCY = ('<CURRENCY>',  '(usd|USD|\$|sgd|SGD)')
TAG_ADDRESS = ('<ADDRESS>', None, None)

TAG_SELF = ("<SELF>", ".*", 99)

# Tag short is for eliminate some bad cases in PERSON, ORGANIZATION and etc.
# TAG_SHORT = (None, None, ['/', "Amt", "AMT", "QTY", "Qty", "TOTAL", "Total", "Tel:", "TEL:", "NO.", "No.", "Adj", "ADJ", "Price", "Tax", "TAX", "SGD", "USD", "sgd", "usd", "AMOUNT", "DEC", "Dec", "NOV", "Nov", "OCT", "Oct", "SEPT", "Sept", "AUG", "Aug", "JULY", "July", "JUNE", "June", "Apr", "APR", "Feb", "FEB", "E. &"])

# ALL_TAG_LIST = [TAG_NUMBER, TAG_DATE, TAG_ADDRESS, TAG_AMOUNT, TAG_CURRENCY, TAG_POSTCODE]
# MCH_TAG_LIST = [TAG_ADDRESS, TAG_AMOUNT, TAG_CURRENCY, TAG_POSTCODE, TAG_DATE, TAG_NUMBER]
# TOP_TAG_LIST = [TAG_NUMBER, TAG_DATE, TAG_AMOUNT]