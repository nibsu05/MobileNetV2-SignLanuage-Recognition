import os
from icrawler.builtin import BingImageCrawler

def crawl_asl_data(output_dir="dataset", max_num=100):
    classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    classes.extend(['space', 'del', 'nothing'])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for class_name in classes:
        print(f"Crawling images for class: {class_name}")
        
        class_dir = os.path.join(output_dir, class_name)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        crawler = BingImageCrawler(storage={'root_dir': class_dir})

        keyword = f"ASL sign language letter {class_name}"
        if class_name == 'space':
            keyword = "ASL sign language space"
        elif class_name == 'del':
            keyword = "ASL sign language delete"
        elif class_name == 'nothing':
            keyword = "ASL sign language nothing"

        crawler.crawl(keyword=keyword, max_num=max_num)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl ASL alphabet images")
    parser.add_argument("--max_num", type=int, default=100, help="Maximum number of images per class")
    args = parser.parse_args()
    
    crawl_asl_data(max_num=args.max_num)
