# %%
from abc import ABC, abstractmethod
import datetime
import os
from pathlib import Path
import httpx
from asyncio import Semaphore
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
import asyncio
from bs4 import BeautifulSoup
import urllib
import json
from typing import List, Dict, Optional
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

from hungarian_peter_wordcloud import DATA_DIR


class BaseFetcher(ABC):  # Inherit from ABC (Abstract Base Class)
    def __init__(
        self, base_url: str, timeout: int = 10, max_connections: int = 3
    ) -> None:
        """
        Initialize the BaseFetcher with a base URL, timeout, and maximum connections.

        :param base_url: The base URL for fetching articles.
        :param timeout: The timeout for HTTP requests.
        :param max_connections: The maximum number of concurrent connections.
        """
        self.base_url = base_url
        self.article_links: List[str] = []
        self.articles: List[Dict[str, Optional[str]]] = []
        self.timeout = timeout
        self.semaphore = Semaphore(max_connections)

    @abstractmethod
    def retrieve_links(self, page: int) -> None:
        """
        Retrieve article links from a given page.
        The links must be stored in the article_links attribute as a flat list of URLs.

        :param page: The page number to retrieve links from.
        """
        pass

    @abstractmethod
    def parse_article(
        self, article_html: str, article_url: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        Parse a single article's HTML.

        :param article_html: The HTML content of the article.
        :param article_url: The URL of the article.
        :return: A dictionary containing parsed article data.
        """
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def fetch_article(
        self, client: httpx.AsyncClient, article_url: str
    ) -> Optional[str]:
        """
        Fetch a single article with retry logic.

        :param client: The HTTP client to use for fetching.
        :param article_url: The URL of the article to fetch.
        :return: The HTML content of the article, or None if fetching failed.
        """
        try:
            response = await client.get(article_url, timeout=self.timeout)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.warning(f"Failed to fetch {article_url}: {e}")
            logger.warning(f"Retrying {article_url}...")
            raise e  # Raise exception to trigger retry

    async def fetch_all_articles(self, pages: int) -> List[Optional[str]]:
        """
        Fetch all articles across multiple pages.

        :param pages: The number of pages to fetch articles from.
        :return: A list of HTML content for each article.
        """
        for page in tqdm(range(1, pages + 1), desc="Retrieving Links..."):
            try:
                self.retrieve_links(page)
            except Exception as e:
                logger.warning(f"Failed to retrieve links from page {page}: {e}")
                raise e

        async with httpx.AsyncClient() as client:
            tasks = [
                self.fetch_article(client, article_url)
                for article_url in self.article_links
            ]
            wrapped_tasks = tqdm_asyncio.gather(*tasks, desc="Fetching Articles...")
            return await wrapped_tasks

    async def run_async(self, pages: int = 1) -> None:
        responses = await self.fetch_all_articles(pages)
        self.articles = [
            self.parse_article(article_html, article_url)
            for article_html, article_url in tqdm(
                zip(responses, self.article_links),
                desc="Parsing Articles...",
                total=len(self.article_links),
            )
            if article_html is not None
        ]


class MNOFetcher(BaseFetcher):
    def __init__(self, timeout: int = 10, max_connections: int = 3) -> None:
        """
        Initialize the MNOFetcher with a specific base URL, timeout, and maximum connections.

        :param timeout: The timeout for HTTP requests.
        :param max_connections: The maximum number of concurrent connections.
        """
        super().__init__(
            "https://magyarnemzet.hu/cimke/magyar-peter",
            timeout=timeout,
            max_connections=max_connections,
        )

    def retrieve_links(self, page: int) -> None:
        """
        Retrieve links from a page. They are stored in the article_links attribute.

        :param page: The page number to retrieve links from.
        """
        page_url = f"{self.base_url}?page={page}"
        response = httpx.get(page_url)
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", class_="article-link-wrapper")
        base_url = "https://magyarnemzet.hu"
        self.article_links.extend(base_url + link.get("href") for link in links)

    def parse_article(
        self, article_html: str, article_url: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        Parse an article.

        :param article_html: The HTML content of the article.
        :param article_url: The URL of the article.
        :return: A dictionary containing parsed article data.

        Example usage:
        >>> fetcher = MNOFetcher()
        >>> asyncio.run(fetcher.run_async(pages=1))
        """
        try:
            soup = BeautifulSoup(article_html, "html.parser")
            article_text_block = soup.find("app-article-text", class_="is-first")
            if not article_text_block:
                logger.warning(f"Article text block not found in {article_url}")
                return None

            paragraphs = article_text_block.find_all(["p", "li"])
            article_text = "\n".join(
                [
                    "- " + para.get_text(separator="\n", strip=True)
                    if para.name == "li"
                    else para.get_text(separator="\n", strip=True)
                    for para in paragraphs
                ]
            )

            def safe_find_text(selector, class_name):
                element = soup.find(selector, class_=class_name)
                return element.get_text(strip=True) if element else None

            title = safe_find_text("h1", "title")
            lead = safe_find_text("h2", "lead")
            publish_date = safe_find_text("span", "publishdate")
            reference_source = safe_find_text("span", "source")
            if reference_source:
                reference_source = reference_source.replace("Forrás:", "").strip()

            return {
                "source": "Magyar Nemzet",
                "title": title,
                "lead": lead,
                "publish_date": publish_date,
                "reference_source": reference_source,
                "author_name": None,
                "article_text": article_text,
                "url": article_url,
            }
        except Exception as e:
            logger.warning(f"Failed to parse {article_url}: {e}")
            return None


class TelexFetcher(BaseFetcher):
    def __init__(self, timeout: int = 10, max_connections: int = 3) -> None:
        """
        Initialize the TelexFetcher with a specific base URL, timeout, and maximum connections.

        :param timeout: The timeout for HTTP requests.
        :param max_connections: The maximum number of concurrent connections.

        Example usage:
        >>> fetcher = TelexFetcher()
        >>> asyncio.run(fetcher.run_async(pages=1))

        """
        super().__init__(
            "https://telex.hu/archivum",
            timeout=timeout,
            max_connections=max_connections,
        )

    def retrieve_links(self, page: int) -> None:
        """
        Retrieve links from a page.

        :param page: The page number to retrieve links from.
        """
        # Telex uses a custom URL encoding scheme, passing the filter object as a JSON string. We need to convert it back.
        filters = {
            "and_tags": ["Magyar Péter"],
            "superTags": [],
            "authors": [],
            "title": [],
        }

        filters_str = json.dumps(filters, ensure_ascii=False)

        query_params = {
            "term": "",
            "filters": filters_str,
            "oldal": page,
        }

        encoded_query = urllib.parse.urlencode(
            query_params, quote_via=urllib.parse.quote
        )

        response = httpx.get("https://telex.hu/archivum", params=encoded_query)

        soup = BeautifulSoup(response, "html.parser")

        list_group = soup.find("div", class_="list__group")
        links = list_group.find_all("a", class_="list__item__title")

        base_url = "https://telex.hu"
        self.article_links.extend(base_url + link.get("href") for link in links)

    def parse_article(
        self, article_html: str, article_url: str
    ) -> Optional[Dict[str, Optional[str]]]:
        """
        Parse an article.

        :param article_html: The HTML content of the article.
        :param article_url: The URL of the article.
        :return: A dictionary containing parsed article data.
        """
        try:
            soup = BeautifulSoup(article_html, "html.parser")
            article_text_block = soup.find("div", class_="article-html-content")
            if not article_text_block:
                logger.warning(f"Article text block not found in {article_url}")
                return None

            paragraphs = article_text_block.find_all(["p", "li"])
            article_text = "\n".join(
                [
                    "- " + para.get_text(separator="\n", strip=True)
                    if para.name == "li"
                    else para.get_text(separator="\n", strip=True)
                    for para in paragraphs
                ]
            )

            def safe_find_text(selector=None, class_name=None):
                element = soup.find(selector, class_=class_name)
                return element.get_text(strip=True) if element else None

            title = safe_find_text("h1", None)
            publish_date = safe_find_text("div", "history content-wrapper__child")
            author_name = safe_find_text("a", "author__name")

            return {
                "source": "Telex",
                "title": title,
                "lead": None,
                "publish_date": publish_date,
                "reference_source": None,
                "author_name": author_name,
                "article_text": article_text,
                "url": article_url,
            }
        except Exception as e:
            logger.warning(f"Failed to parse {article_url}: {e}")
            return None


async def collect_articles(
    *fetchers: List[BaseFetcher],
    pages: int = 1,
) -> List[Dict[str, Optional[str]]]:
    if not fetchers:
        logger.warning("No fetchers provided. Defaulting to MNO and Telex.")
        fetchers = [MNOFetcher(max_connections=3), TelexFetcher(max_connections=3)]
    for fetcher in fetchers:
        await fetcher.run_async(pages=pages)
    logger.success("Articles collected successfully.")
    logger.info(f"Articles collected from {len(fetchers)} sources.")
    logger.info(
        f"Total articles collected: {sum(len(fetcher.articles) for fetcher in fetchers)}"
    )
    logger.info(
        f"Total articles parsed: {sum(1 for fetcher in fetchers for article in fetcher.articles if article is not None)}"
    )
    logger.info(
        f"Total articles with content: {sum(1 for fetcher in fetchers for article in fetcher.articles if article['article_text'] is not None)}"
    )
    return [article for fetcher in fetchers for article in fetcher.articles]


def save_articles(articles: List[Dict[str, Optional[str]]], filename: Path) -> None:
    with open(filename.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    logger.success(f"Articles saved successfully to {Path(filename).resolve()}.")


def main(pages: int = 1) -> None:
    raw_articles_directory = DATA_DIR / "articles_raw"
    raw_articles_directory.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fetchers = [MNOFetcher(max_connections=3), TelexFetcher(max_connections=3)]
    articles = asyncio.run(collect_articles(*fetchers, pages=pages))
    save_articles(articles, raw_articles_directory / f"articles_{date}")


# %% Downloading articles from MNO and Telex
if __name__ == "__main__":
    main(pages=10)
    # %% Example notebook usage. Notebook only accepts await, it has internal async loop handling.
    # await collect_articles(pages=1)
