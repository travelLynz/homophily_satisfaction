import pandas as pd
import settings as s
import string

def get_country_dic(countries):
    c_dic = {}
    for _, r in countries.iterrows():
        c_dic[r['name']] = r['alpha2Code']
        if r['name']  != r['nativeName']:
            c_dic[r['nativeName']] = r['alpha2Code']
    return c_dic

def get_city_info(vals, countries):
  country, ccode, city, state = [], [], [], []
  for v in vals:
      split_v = [i.replace('\u200b\u200b', '').strip() for i in v.split(",")]
      if len(split_v) <= 1:
          if v in countries.keys():
              country.append(v)
              ccode.append(countries[v])
              city.append("Unknown")
              state.append("Unknown")
          elif len(v.split(" ")) > 1 and v.split(" ")[-1] in countries.keys():
              s_v = v.split(" ")
              country.append(s_v[-1])
              ccode.append(countries[s_v[-1]])
              city.append(s_v[:-1])
              state.append("Unknown")
          elif v in s.states:
              country.append("United States")
              ccode.append("US")
              city.append("Unknown")
              state.append(v)
          else:
              country.append(v)
              ccode.append("UNK")
              city.append("Unknown")
              state.append("Unknown")
      elif split_v[-1] in countries.keys():
          country.append(split_v[-1])
          ccode.append(countries[split_v[-1]])
          city.append(split_v[0])
          state.append("Unknown")
      elif len(split_v[1].split(" ")) > 1 and split_v[1].split(" ")[1] in countries.keys():
          #print(split_v)
          country.append(split_v[1].split(" ")[1])
          ccode.append(countries[split_v[1].split(" ")[1]])
          city.append(split_v[0])
          state.append("Unknown")
      elif split_v[-1] in s.states:
          country.append("United States")
          ccode.append("US")
          city.append(split_v[0])
          state.append(split_v[-1])
      else:
          #print(split_v)
          country.append(split_v[1])
          ccode.append("UNK")
          city.append(split_v[0])
          state.append("Unknown")
  return (country, ccode, city, state)

def add_additional_c_matches(countries):
    countries["Russia"] = "RU"
    countries["Vietnam"] = "VN"
    countries["South Korea"] = "KR"
    countries["UK"] = "GB"
    countries["USA"] = "US"
    countries["US Virgin Islands"] = "VI"
    countries["Hong-Kong"] = "HK"
    countries["Dutch Caribbean"] = "BQ"
    countries["Saint Martin"] = "SX"
    countries["Argentine"] = "AR"
    countries["New Zeland"] = "NZ"
    countries["Aland Islands"] = "AX"
    countries["Reunion"] = "RE"
    countries["Laos"] = "LA"
    return countries

def get_country_info():
    countries = pd.read_json('https://restcountries.eu/rest/v2/all')[['alpha2Code','alpha3Code' , 'languages', 'borders', 'capital', 'gini', 'latlng', 'name', 'nativeName', 'population', 'region', 'regionalBlocs', 'subregion']]
    countries.regionalBlocs = countries.regionalBlocs.map(lambda x: [i['acronym'] for i in x])
    countries.languages = countries.languages.map(lambda x: [i['name'] for i in x])
    return countries

def extract_ccode(tbl, col, c_dic, ic_dic):
    vals = tbl[col].map(lambda x: x.replace('"', '').replace("\'", '').split(',') if x != None else x)
    tbl['ccode'] = [find_country(v, c_dic, ic_dic) for v in vals]
    return tbl

def find_country(vals, cd, icd):
    results = []
    for v in vals:
        v = v.replace('\u200b\u200b', '').strip()
        if v in cd.keys():
            results.append(cd[v])
        elif v in icd:
            results.append(v)
        elif string.capwords(v) in cd.keys():
            results.append(cd[string.capwords(v)])
        elif v in s.states:
            results.append("US")
        else:
            sp =v.split(' ')
            if sp[-1] in cd.keys():
                results.append(cd[sp[-1]])
    return results[-1] if len(results) > 0 else None
