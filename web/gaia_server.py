import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

from bigscience_pii_detect_redact import run_pii
from pyserini.analysis import Analyzer
from pyserini.search.lucene import LuceneSearcher


MAX_DOCS = 5

index_dirs = {
    "laion": "/home/piktus_huggingface_co/index/laion2B-en-index/dedup/",
    "c4": "/home/piktus_huggingface_co/index/segmented-c4-en-index/",
    "pile": "/home/piktus_huggingface_co/index/dedup_pile_indices/",
}


class ThreadedPyseriniHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)

        logging.info("initializing lucene")
        self.searcher = {}
        self.analyzer = {}

        for corpus, index_dir in index_dirs.items():
            self.searcher[corpus] = LuceneSearcher(index_dir)
            self.analyzer[corpus] = Analyzer(self.searcher[corpus].object.analyzer)


class PyseriniHTTPRequestHandler(BaseHTTPRequestHandler):
    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

    def _process_hits(self, hits, corpus, query_terms, highlight_terms=None):
        hits_entries = []
        if highlight_terms is None:
            highlight_terms = set()
        for hit in hits:
            raw = json.loads(hit.raw)
            hits_entry = {}
            hits_entry["docid"] = hit.docid
            hits_entry["score"] = hit.score
            hits_entry["text"] = raw["contents"]
            if "meta" in raw:
                hits_entry["meta"] = raw["meta"]
            piis = run_pii(hits_entry["text"], None)
            if len(piis[1]) > 0:
                hits_entry["text"] = piis[1]["redacted"]
            hits_entries.append(hits_entry)
            hit_terms = hits_entry["text"].split()
            for term in hit_terms:
                term_rewritten = set(self.server.analyzer[corpus].analyze(term))
                if len(query_terms & term_rewritten) > 0:
                    highlight_terms.add(term)
        logging.info("Highlight terms: {}".format(str(highlight_terms)))
        return hits_entries, highlight_terms

    def do_GET(self):
        logging.info(
            "GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers)
        )
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode("utf-8"))

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length).decode("utf-8")
        logging.info(
            "POST request,\nPath: {}\nHeaders:\n{}\nBody:\n{}\n".format(
                self.path, self.headers, post_data
            )
        )

        post_data = json.loads(post_data)
        if "flag" in post_data and bool(post_data["flag"]):
            # TODO: improve reporting
            self._set_response()
            self.wfile.write(json.dumps("Flagging OK").encode("utf-8"))
            return

        query = post_data["query"]
        corpus = post_data["corpus"]

        k = (
            MAX_DOCS
            if "k" not in post_data or post_data["k"] is None
            else int(post_data["k"])
        )

        logging.info("Query: {}".format(query))

        results = None
        query_terms = set(self.server.analyzer[corpus].analyze(query))
        results, highlight_terms = self._process_hits(
            self.server.searcher[corpus].search(query, k=k), corpus, query_terms
        )

        payload = {"results": results, "highlight_terms": list(highlight_terms)}
        self._set_response()
        self.wfile.write(json.dumps(payload).encode("utf-8"))


def run(server_address, port):
    logging.basicConfig(level=logging.INFO)
    httpd = ThreadedPyseriniHTTPServer(
        (server_address, port), PyseriniHTTPRequestHandler
    )

    sa = httpd.socket.getsockname()
    logging.info("Starting httpd on {} port {} ...".format(sa[0], sa[1]))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--server_address",
        required=True,
        type=str,
        help="Address of the server, e.g. '12.345.678.910'",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8080,
        help="Port on which to serve ",
    )
    args = parser.parse_args()
    run(args.server_address, args.port)
