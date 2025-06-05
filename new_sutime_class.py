import jpype
import jpype.imports
from jpype.types import *
import csv
import re
import json
import os
from tqdm import tqdm
import pandas as pd

corenlp_dir = "C:/Users/user/OneDrive/Desktop/data_NER/sutime/sutimetest/stanford-corenlp-4.5.7"
classpath = os.path.join(corenlp_dir, "*")
if not jpype.isJVMStarted():
    jpype.startJVM(classpath=[classpath])
    
defs_sutime_path = "C:/Users/user/OneDrive/Desktop/data_NER/sutime/sutimetest/defs.sutime.txt"
english_sutime_path = "C:/Users/user/OneDrive/Desktop/data_NER/sutime/sutimetest/english.sutime.txt"

from edu.stanford.nlp.pipeline import StanfordCoreNLP, CoreDocument
from edu.stanford.nlp.time import TimeAnnotations
from java.util import Properties

class SUTimeExtractor:
    def __init__(self, defs_sutime_path, english_sutime_path):
        self.defs_sutime_path = defs_sutime_path
        self.english_sutime_path = english_sutime_path
        self.pipeline = None
        
        self.setup_pipeline()
    def setup_pipeline(self):
        # Set up pipeline properties
        props = jpype.JClass('java.util.Properties')()
        props.setProperty("annotators", "tokenize,pos,lemma,ner")
        props.setProperty("ner.docdate.usePresent", "true")
        props.setProperty("sutime.includeRange", "true")
        props.setProperty("ner.rulesOnly", "true")
        props.setProperty("sutime.rules", f"{self.defs_sutime_path},{self.english_sutime_path}")
        props.setProperty("sutime.markTimeRanges", "true")

        # Build pipeline
        self.pipeline = StanfordCoreNLP(props)

    def read_csv(self, path):
        with open(path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            return list(reader)

    def process_document(self, text, doc_id):
        document = CoreDocument(text)
        self.pipeline.annotate(document)

        captured_expressions = set()
        captured_expressions.add(None)
        captured_expressions.add("")
        temporal_data = TemporalData(doc_id)

        for cem in document.entityMentions():
            timex = cem.coreMap().get(TimeAnnotations.TimexAnnotation)
            if timex:
                original_input = str(cem.text())
                print("Original input:", original_input)

                # Matches "2023-07-03 12:00", "2024.04.07 3:13", "2024/4/7 12:34", "2024.04.07 03:13:29"
                datetime_pattern = re.compile(
                         r"(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2})\s*(\d{1,2}:\d{1,2}(?::\d{1,2})?)?"  
                )
                datetime_matcher = datetime_pattern.search(original_input)

                # Regex for datetime format "2:00 5/9 or 2:00 5/9/2024 or 2:00:00 5/9 or 2:00:00 5/9/2024"
                specific_datetime_pattern = re.compile(
                    r"(\d{1,2}:\d{1,2}(?::\d{1,2})?)\s*(\d{1,2}[\/\-\.]\d{1,2})(?:[\/\-\.](\d{2,4}))?"
                )
                specific_datetime_matcher = specific_datetime_pattern.search(original_input)

                # Regex for datetime format "2:00 2 tháng 2 năm 2023"
                new_datetime_pattern = re.compile(r"(\d{1,2}:\d{2})\s*(\d{1,2})\s*tháng\s*(\d{1,2})\s*năm\s*(\d{4})")
                new_datetime_matcher = new_datetime_pattern.search(original_input)

                # Regex for time formats like "12h", "16h", "16 giờ"
                time_pattern = re.compile(
                            r"\b(\d{1,2}:\d{2}(?::\d{2})?)\b"  # Captures time like "14:30" or "14:30:15"
                            r"|\b(\d{1,2}\s*h(?:\s*\d{1,2})?)\b"      # Captures time like "14h30"
                            r"|\b(\d{1,2}r)\b"    
                            r"|\b(\d{1,2}\s*giờ(?:\s*\d{1,2}(?:\s*phút)?)?)\b"  # Captures "7 giờ", "7 giờ 30", "7 giờ 30 phút"
                            r"|\b(\d{1,2}\s*rưỡi)\b"  # Captures "10 rưỡi"
                        )
                time_matcher = time_pattern.finditer(original_input)

                # Regex for date formats like "27 tháng 03" or "tháng 1/2020"
                """
                r"(năm\s*\d{4})|"  # Matches "năm 2019"
                r"(Năm\s*\d{4})|"  # Matches "Năm 2019"
                r"\b(\d{4})-(\d{4})\b|"  # Matches year ranges like "2024-2025"
                r"tháng\s*(\d{1,2})[\/\-.](\d{4})|"  # Matches "tháng 1/2020"
                r"(\d{1,2})\s*tháng\s*(\d{1,2})(?:\s*năm\s*(\d{4}))?|"  # Matches "27 tháng 03 năm 2023"
                r"(\d{1,2}[\/\-]\d{1,2}(?:[\/\-]\d{4})?)|"  # Matches "5/9/2023", "5-9-2023"
                r"(\d{4}-\d{2}-\d{2})"  # Matches "2023-07-03"
                """
                
                date_pattern = re.compile(
                        r"\b(\d{4}[./-]\d{1,2}[./-]\d{1,2})\b|"  # Matches "2024.05.28", "2024/05/28", "2024-05-28"
                        r"\bngày\s+(\d{1,2}[\/\-\.]\d{1,2})\b|"  # Matches "ngày 30/09", "ngày 13/03"
                        r"\bngày\s+(\d{1,2}\s*tháng\s*\d{1,2}(?:\s*năm\s*\d{4})?)\b|"  # Matches "ngày 12 tháng 11 năm 2023"
                        r"\b(\d{1,2}[\/\-\.]\d{1,2}(?:[\/\-\.]\d{2,4})?)\b|"  # Matches "30/09", "30-09", "30/09/2024"
                        r"\b(\d{1,2}\s*tháng\s*\d{1,2}(?:\s*năm\s*\d{4})?)\b|"  # Matches "2 tháng 11 năm 2023", "12 tháng 11 năm 2023"
                        r"\bngày\s+(\d{1,2})\b"  # Matches "ngày 6", "ngày 31"
                    )
                
                
                date_matcher = date_pattern.finditer(original_input)

                if specific_datetime_matcher:
                    # Handle "2:00 5/9", "2:00 5-9", "2:00 5/9/2024", or "2:00 5-9-2023"
                    time_part = specific_datetime_matcher.group(1)  # "2:00"
                    date_part = specific_datetime_matcher.group(2)  # "5/9", "5-9"
                    year_part = specific_datetime_matcher.group(3)  # "2024", "2023" (optional)

                    # Determine the correct separator based on the date_part
                    separator = '-' if '-' in date_part else '/'
                    separator = '.' if '.' in date_part else separator

                    # Construct full date correctly using the same separator
                    full_date = f"{date_part}{separator}{year_part}" if year_part else date_part

                    if time_part not in captured_expressions:
                        temporal_data.add_detail(time_part, "TIME")
                        captured_expressions.add(time_part)
                    if full_date not in captured_expressions:
                        temporal_data.add_detail(full_date, "DATE")
                        captured_expressions.add(full_date)

                if datetime_matcher:
                    # Handle "2023-07-03 12:00"
                    original_date = datetime_matcher.group(1)  # e.g., "2023-07-03"
                    original_time = datetime_matcher.group(2)  # e.g., "12:00"

                    if original_time not in captured_expressions:
                        temporal_data.add_detail(original_time, "TIME")
                        captured_expressions.add(original_time)
                    if original_date not in captured_expressions:
                        temporal_data.add_detail(original_date, "DATE")
                        captured_expressions.add(original_date)

                if new_datetime_matcher:
                    # Handle "2:00 2 tháng 2 năm 2023"
                    time_expression = new_datetime_matcher.group(1)
                    date_expression = f"{new_datetime_matcher.group(2)} tháng {new_datetime_matcher.group(3)} năm {new_datetime_matcher.group(4)}"

                    if time_expression not in captured_expressions:
                        temporal_data.add_detail(time_expression, "TIME")
                        captured_expressions.add(time_expression)
                    
                    if date_expression not in captured_expressions:
                        temporal_data.add_detail(date_expression, "DATE")
                        captured_expressions.add(date_expression)
            
                # Handle general date and time formats
                for match in time_matcher:
                    time_expression = match.group(0)
                    if time_expression not in captured_expressions:
                        temporal_data.add_detail(time_expression, "TIME")
                        captured_expressions.add(time_expression)

                for match in date_matcher:
                    date_expression = match.group(0)
                    if date_expression not in captured_expressions:
                        temporal_data.add_detail(date_expression, "DATE")
                        captured_expressions.add(date_expression)
                # In case none of the specific patterns match and no other patterns are found
                if not (specific_datetime_matcher or datetime_matcher or new_datetime_matcher or date_matcher or time_matcher):
                    print(1)
                    if original_input not in captured_expressions:
                        print(2)
                        type_pattern = re.compile(r'type="([^"]+)"')
                        match = type_pattern.search(str(timex))
                        if match:
                            extracted_type = match.group(1)
                            type_ = extracted_type
                        else:
                            type_ = "Type not found"
                        temporal_data.add_detail(original_input, type_)
                        captured_expressions.add(original_input)

        return temporal_data

    def process_csv(self, path):
        data = self.read_csv(path)
        temporal_data_map = {}

        for i in tqdm(range(1, len(data))):
            example = data[i]
            temporal_data = self.process_document(example[0], i)
            temporal_data_map[i] = temporal_data

        return temporal_data_map

    def write_to_json(self, temporal_data_map, output_path):
        sorted_data = sorted(temporal_data_map.values(), key=lambda x: x.get_id())
        json_data = [{"id": data.get_id(), "details": data.details} for data in sorted_data]

        with open(output_path, 'w', encoding='utf-8') as writer:
            json.dump(json_data, writer, ensure_ascii=False, indent=4)

    def shutdown_jvm(self):
        if jpype.isJVMStarted():
            jpype.shutdownJVM()
    def postprocess(self, csv_input_path, json_input_path, output_csv_path):
        df = pd.read_csv(csv_input_path)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'id'}, inplace=True)

        with open(json_input_path, 'r', encoding="utf8") as file:
            data = json.load(file)

        tag_list = []
        for i in range(len(data)):
            set_tag = set()
            tag = []
            for j in range(len(data[i]['details'])):
                element_2 = str(data[i]['details'][j]['text'])
                element_1 = str(data[i]['details'][j]['type'])
                set_tag.add((element_1, element_2))
                tag = list(set_tag)
            tag_list.append(tag)

        json_df = pd.DataFrame(data)
        json_df['id'] = json_df['id'] - 1
        json_df['details'] = tag_list

        merged_df = pd.merge(df, json_df, on='id', how='left')
        merged_df = merged_df.drop(columns="id")

        merged_df.to_csv(output_csv_path, index=False)
    

class TemporalData:
    def __init__(self, id):
        self.id = id
        self.details = []

    def add_detail(self, value, label):
        self.details.append({"text": value, "type": label})

    def get_id(self):
        return self.id
def test_sutime_extractor(sutime_extractor, input_text):
    # Process the input text
    temporal_data = sutime_extractor.process_document(input_text, 1) 
    print(f"Input: {input_text}")
    print("Extracted Temporal Data:")
    for detail in temporal_data.details:
        print(f" - {detail['type']}: {detail['text']}")
# Usage example
if __name__ == "__main__":
    sutime_extractor = SUTimeExtractor(defs_sutime_path, english_sutime_path)
    #test_input = "2:00 2/5/2023 và 12:34 2/5, và 2:00 2-5-2023 và 2023-07-03 12:00 và 3-7-2023 13:00 và 2 tháng 2 năm 2023 và 12 tháng 3 và 12.6 và 12.6.2023 và 16 giờ và 16h và 4:30 2.5.2023 và 17:42 2.5 và 16h30 và 16 giờ 30 và ngày 20 và 08:24:30"
    #test_input = "2:00 2/5 Cơ quan điều tra cáo buộc, từ năm 2019 đến 2021, ông Thọ đã lợi dụng chức vụ, quyền hạn được giao để nhận hối lộ và gây ảnh hưởng với người khác để trục lợi. Cụ thể, trong việc Công ty Xuyên Việt Oil vay vốn tại Vietinbank và xin cấp giới hạn tín dụng, kéo dài thời gian duy trì giới hạn tín dụng, từ năm 2019 đến tháng 1/2020, ông Thọ đã hai lần nhận hối lộ, tổng 600.000 USD (tương đương 13,8 tỷ đồng) của Mai Thị Hồng Hạnh."
    #test_input = "ngày 12 tháng 11 năm 2023 và 2024-5-2 9:00 và 2024.05.28 và 29/05 và 29/6 và 16 giờ và 2024.4.7 3:13 và 6/1/23 và 2:00 14/2/25 và ngày 20 và ngày 6/8"
    #test_input = "2024.05.28 và ngày 06 và ngày 30/09 và ngày 31 và 2 tháng 11 năm 2023 và ngày 12 tháng 11 năm 2023 và ngày 13 tháng 3 và 2024-5-2 9:00 và 12.6 và 12.6.2020 và 7/6 và 21 tháng 03 "
    test_input = "7 h và 7h và 7 h 30 và 7 h30 và 7h 30"
    #test_input = "08:21:31 28/8 và 28/9 08:22:30 và 7h50 8/9 và 7 giờ 30 phút và 8 giờ 39 và 8 rưỡi ngày 12 tháng 11 năm 2023 và 5r"
    #test_input = "Ngân hàng tôi sử dụng là MB Bank chủ tài khoản Nguyễn Văn Phước stk : 0857362514 đơn hàng đặt vào 2023-07-03 at 12:00:00."

    test_sutime_extractor(sutime_extractor, test_input)
    
    """
    temporal_data_map = sutime_extractor.process_csv("C:/Users/user/OneDrive/Desktop/data_NER/data_cs/test_cs.csv")
    sutime_extractor.write_to_json(temporal_data_map, "output_test_new_sutime.json")
    sutime_extractor.postprocess("C:/Users/user/OneDrive/Desktop/data_NER/data_cs/test_cs.csv",
                          "C:/Users/user/OneDrive/Desktop/data_NER/output_test_new_sutime.json",
                          "eval_sutime_test_new.csv")
    
    """
    sutime_extractor.shutdown_jvm()
