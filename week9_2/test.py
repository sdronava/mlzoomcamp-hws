import json

from lambda_function import lambda_handler


def test_lambda_handler():
    #https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg
    event = {  "url": "https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg" }
    context = None
    result = lambda_handler(event, context)
    #print(result)

test_lambda_handler()
