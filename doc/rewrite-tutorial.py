# coding: utf-8

# In[41]:

from bs4 import BeautifulSoup
import glob2
from collections import OrderedDict

import sys, os
from tutorial_dir_title_conf import dir_title_config

args = sys.argv

language = str(args[1])

notebooks_dir = "_build/html/notebooks/"

# tutorial files pattern
tutorials_pattern = "_build/html/notebooks/**/notebook.html"

# tutorial list file
target_pattern = "_build/html/notebooks/tutorials.html"

# tutorial0-template/notebook.html full path
tutorial_template = "_build/html/notebooks/tutorial0-template/notebook.html"

category_files = OrderedDict()

# Reverse and Roop
for v in dir_title_config:

    temp_dic = {}

    if language == "ja":
        temp_dic["title"] = v[1]
    elif language == "en":
        temp_dic["title"] = v[2]

    category_files_path = notebooks_dir + str(v[0]) + "/**/*.html"

    files = glob2.glob(category_files_path)

    temp_dic["files"] = files
    category_files[v[0]] = temp_dic

# Extract excerptions from each tutorial and edit tutorials.html
target_soup = BeautifulSoup(open(target_pattern), "html.parser")

# create  tutorial_lists_wrapper
# <ul class="tutorial_lists_wrapper"> ~ </ul>
tutorial_lists_wrapper = target_soup.find("div", {"id": "tutorial"}).find("div", {"class": "toctree-wrapper"}).find(
    "ul")
tutorial_lists_wrapper.clear()
tutorial_lists_wrapper["class"] = "tutorial_lists_wrapper"

for k, v in category_files.items():
    category_name = v["title"]

    # create tutorial_lists tag
    # <ul class="tutorial_lists"> ~ </ul>
    tutorial_lists = target_soup.new_tag("ul")
    tutorial_lists["class"] = "tutorial_category_lists"

    # create tutorial_category_list
    # <li> ~ </li>
    tutorial_category_list = target_soup.new_tag("li")
    tutorial_category_list["class"] = "tutorial_category_list_item"

    # create category name tag
    # <h2 class="category_name> ~ </2>
    category_name_tag = target_soup.new_tag("h2")
    category_name_tag["class"] = "category_name"
    category_name_tag.append(category_name)

    tutorial_category_list.append(category_name_tag)

    # create tutorial list tag
    tutorial_list = target_soup.new_tag("ul")
    tutorial_list["class"] = "tutorial_lists"

    for html_file in v["files"]:

        soup = BeautifulSoup(open(html_file), "html.parser")

        summary_text = soup.find("div", {"class": "document"}).find("div", {"itemprop": "articleBody"}).select(
            " > div.section  .summary")[0].get_text()

        title_text = soup.find("h1").text
        if ("¶" in title_text):
            title_text = title_text.replace("¶", "")

        link_href = html_file.split("/")
        link_href = link_href[-3] + "/" + link_href[-2] + "/" + link_href[-1]

        # <li class="tutorial_list_item"> ~ </li>
        tutorial_list_item = target_soup.new_tag("li")
        tutorial_list_item["class"] = "tutorial_list_item"

        # tutorial link tag 
        # <a> ~ </a>
        tutorial_link = target_soup.new_tag("a")
        tutorial_link["href"] = link_href

        # tutorial title tag 
        # <h3> ~ </h3>
        tutorial_title = target_soup.new_tag("h3")
        tutorial_title["class"] = "tutorial_title"
        tutorial_title.append(title_text)

        # tutorial summary tag 
        # <p class="summary"> ~ </sumamry>
        summary_tag = target_soup.new_tag("p")
        summary_tag.append(summary_text)
        summary_tag["class"] = "summary"

        # append to tutorial link tag
        tutorial_link.append(tutorial_title)
        tutorial_link.append(summary_tag)

        # append link tag to tutorial list item
        tutorial_list_item.append(tutorial_link)

        tutorial_list.append(tutorial_list_item)

    tutorial_category_list.append(tutorial_list)
    tutorial_lists_wrapper.append(tutorial_category_list)



with open(target_pattern, "w") as ftpr:
    ftpr.write(target_soup.prettify())
