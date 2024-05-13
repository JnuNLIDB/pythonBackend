import json
import os
import sys

if __name__ == '__main__':
    os.mkdir("data")
    with open("data/theory_preprocessed.txt", "w", encoding="utf-8") as f:
        for line in open(sys.argv[1], "r", encoding="utf-8"):
            j = json.loads(line)
            content = "".join(j["Content"]).replace(" ", "").replace("写材料，就上公文思享文库www.gwsxwenku.com", "")
            source = j["Source"].replace("——", "")
            f.write(source)
            f.write("\n")
            f.write(content)
            f.write("\n\n")
