from bs4 import BeautifulSoup
import glob2
import re


html_files = glob2.glob("_build/html/**/*.html")
match = re.compile('.*jsMath.*')

for html in html_files:

    soup = BeautifulSoup(open(html), "html.parser")

    paths = soup.find_all("script", {"src": match})
    if(len(paths) > 0):
        for path in paths:
            src = "/_static/jsMath-3.6e/easy/load.js"
            path["src"] = src

        with open(html, "w") as ftpr:
            ftpr.write(soup.prettify())

    results = soup.find_all("a", text="old-versions")
    if len(results) > 0:
        for result in results:
            src = result.get("href")
            src = "/old-versions"

            result["href"] = src

        with open(html, "w") as ftpr:
            ftpr.write(soup.prettify())
