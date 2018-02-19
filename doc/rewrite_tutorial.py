# coding: utf-8
from bs4 import BeautifulSoup
import glob2
from collections import OrderedDict

import sys, os
import re

from tutorial_dir_title_conf import dir_title_config

args = sys.argv
language = str(args[1])

notebooks_dir = "_build/html/notebooks/"

# tutorial files pattern
tutorials_pattern = "_build/html/notebooks/**/notebook.html"

# tutorial list file
tutorial_top = "_build/html/notebooks/catalog.html"

# tutorial0-template/notebook.html full path
tutorial_template = "_build/html/notebooks/tutorial0-template/notebook.html"

index_template = "_build/html/index.html"

tutorial_top_soup = BeautifulSoup(open(tutorial_top), "html.parser")

sidebar_soup = tutorial_top_soup.find("nav", {"class": "wy-nav-side"})

tutorial_file_list = []

for a in sidebar_soup.find_all("a"):
    if "notebook.html" == a["href"].split("/")[-1]:
        tutorial_file_list.append(a["href"])

notebook_dir_prefix = '_build/html/notebooks/'

category_files = OrderedDict()

# Reverse and Roop
for v in dir_title_config:

    temp_dic = {}
    temp_files = []

    if language == "ja":
        temp_dic["title"] = v[1]
    elif language == "en":
        temp_dic["title"] = v[2]

    for f in tutorial_file_list:
        if v[0] == f.split("/")[0]:
            temp_files.append(os.path.join(notebook_dir_prefix, f))

    temp_dic["files"] = temp_files
    category_files[v[0]] = temp_dic

# Extract excerptions from each tutorial and edit tutorials.html
tutorial_soup = BeautifulSoup(open(tutorial_top), "html.parser")

# create  tutorial_lists_wrapper
# <ul class="tutorial_lists_wrapper"> ~ </ul>
tutorial_lists_wrapper = tutorial_soup.find("div", {"id": "catalog"}).find("div", {"class": "toctree-wrapper"}).find(
    "ul")
tutorial_lists_wrapper.clear()
tutorial_lists_wrapper["id"] = "tutorial_lists_wrapper"

for k, v in category_files.items():
    category_name = v["title"]

    # create tutorial_lists tag
    # <ul class="tutorial_lists"> ~ </ul>
    tutorial_lists = tutorial_soup.new_tag("ul")
    tutorial_lists["class"] = "tutorial_category_lists"

    # create tutorial_category_list
    # <li> ~ </li>
    tutorial_category_list = tutorial_soup.new_tag("li")
    tutorial_category_list["class"] = "tutorial_category_list_item"

    # create category name tag
    # <h2 class="category_name> ~ </2>
    category_name_tag = tutorial_soup.new_tag("h2")
    category_name_tag["class"] = "category_name"
    category_name_tag.append(category_name)

    tutorial_category_list.append(category_name_tag)

    # create tutorial list tag
    tutorial_list = tutorial_soup.new_tag("ul")
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
        tutorial_list_item = tutorial_soup.new_tag("li")
        tutorial_list_item["class"] = "tutorial_list_item"

        # tutorial link tag
        # <a> ~ </a>
        tutorial_link = tutorial_soup.new_tag("a")
        tutorial_link["href"] = link_href

        # tutorial title tag
        # <h3> ~ </h3>
        tutorial_title = tutorial_soup.new_tag("h3")
        tutorial_title["class"] = "tutorial_title"
        tutorial_title.append(title_text)

        # tutorial summary tag
        # <p class="summary"> ~ </sumamry>
        summary_tag = tutorial_soup.new_tag("p")
        summary_tag.append(summary_text)
        summary_tag["class"] = "summary"

        # company logo
        company_logo = soup.find("div", {"class": "single-company-logo"})

        if company_logo:
            company_logo_img_src = company_logo.find("img").get("src")

            company_logo_tag = tutorial_soup.new_tag("figure")
            company_logo_tag["class"] = "catalog-company-logo"

            company_logo_img = tutorial_soup.new_tag("img")
            company_logo_img["src"] = company_logo_img_src

            company_logo_tag.append(company_logo_img)

        # append to tutorial link tag
        tutorial_link.append(tutorial_title)
        tutorial_link.append(summary_tag)

        if company_logo:
            tutorial_link.append(company_logo_tag)

        # append link tag to tutorial list item
        tutorial_list_item.append(tutorial_link)

        tutorial_list.append(tutorial_list_item)

        document = soup.find("div", {"class": "document"})
        document["class"] = "document single-document " + language + "-document"

        with open(html_file, "w") as ftpr:
            ftpr.write(soup.prettify())

    tutorial_category_list.append(tutorial_list)
    tutorial_lists_wrapper.append(tutorial_category_list)

with open(tutorial_top, "w") as ftpr:
    ftpr.write(tutorial_soup.prettify())

# ### Change index page to tutorial page

index_soup = BeautifulSoup(open(index_template), "html.parser")

tutorial_title = tutorial_soup.find("h1").get_text()

tutorial_title = re.sub(r'\n|\s|¶', "", tutorial_title)

index_soup.find("h1").get_text()

index_soup.find("h1").string = tutorial_title

for a in tutorial_lists_wrapper.find_all("a"):
    a["href"] = os.path.join("notebooks", a["href"])

index_soup.find("div", {"itemprop": "articleBody"}).find("div").find("div", {"class": "toctree-wrapper"}).find(
    "ul").replace_with(tutorial_lists_wrapper)

with open(index_template, "w") as ftpr:
    ftpr.write(index_soup.prettify())
