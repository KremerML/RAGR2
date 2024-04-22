from uplink import Consumer, get, install, loads, patch, put, Path, Query, Header, Body, post, headers, returns, form_url_encoded, response_handler, json, delete, Field, converters
import json

def raise_for_status(response):
    if not response.ok:
        raise ConnectionError('HTTP %s: %s'%(response.status_code,response.text))
    else:
        return response

def htmlize(result):
    title = result['title'][0] if result['title'] else ''
    html = '''
    <li><b>%s</b><br>%s; %s; %s</li>
    '''%(title, result['publishedDate'], ','.join(result['personLastName']), result['doi'])
    return html.strip()

class UTF8Serializer(converters.Factory):

    def create_request_body_converter(self, cls, request_definition):
        return lambda body: body

@response_handler(raise_for_status)
class SSS(Consumer):

    def __init__(self, base_url):
        super().__init__(base_url=base_url, converter=UTF8Serializer())

    @headers('Content-Type: application/json')
    @post('/sharedsearch/document/result')
    def query(self, query: Body,
              product: Header('x-els-product') = 'embeddings',
              SearchDataset: Header('x-els-dataset') = 'embeddings',
              format: Header('accept') = 'application/json'):
        pass

    def simple_query(self, query,freshness,year, k, fields=['all'],
    return_fields=['title', 'publishedDate', 'doi'],
    sort_field='relevance', html=False, product='embeddings'):
        query = {
          "query": {
            "semanticQueryString": query,
            "fields": fields,
#             "defaultOperator": "AND"
          },
            "rankingModel": {
                "modelName": "vectorFreshness",
                "freshness": {
                    "freshnessFactor":freshness,
                    "scriptName":"knn_freshness_script",
                    "queryWeight":1,
                    "rescoreQueryWeight":1
                }
            },
            "filters": [
                     {
                        "fieldName": "subtype",
                        "terms": [
                            "ar",
                            "ch",
                            "bk",
                            "re",
                            "sh",
                            "dp",
                            "cp"
                        ],
                        "operator": "IS ONE OF"
                    },
                     {
                        "fieldName": "pubyyyymm",
                        "range": {
                            "gt": year
                        },
                        "operator": "IS BETWEEN"
                    }
                ],

          "returnFields": return_fields,
          "resultSet": {
            "skip": 0,
            "amount": k
          },
          "sortBy": [
                {
                    "fieldName": "relevance",
                    "order": "desc"
                }
            ],
          "stemming": True
        }
        response = self.query(json.dumps(query), product=product)
        if html:
            results = json.loads(response.content)['hits']
            return html.HTML('<ol>' + '\n'.join(htmlize(result) for result in results) + '</ol>')
        else:
            return response




sss = SSS(base_url='https://shared-search-service-api.staging.scopussearch.net/')

def search_client(query, year=201301,k=20, freshness=0.05):
    r = sss.simple_query(query,freshness,year,k,
        fields=['abs_vector'],
        return_fields=['itemtitle', 'abs','relevance','pubyr','numcitedby']
                         
#         product='scopus'
    )
    return json.loads(r.content)





1
