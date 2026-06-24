"""
rag.py — PostgreSQL + pgvector RAG engine
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
import unicodedata
import math
import urllib.parse
import urllib.request
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple
from urllib.parse import urljoin, urlparse, unquote

import config
import psycopg2

logger = logging.getLogger(__name__)

_embedder: Optional["SentenceTransformer"] = None
_conn = None


# ─────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────

def get_chroma_collection():
    """Kept for backward compatibility — returns a PostgreSQL connection."""
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(
            dbname="ragdb",
            user="raguser",
            password="ragpass",
            host="localhost",
            port=5432,
        )
    return _conn


# ─────────────────────────────────────────────
# EMBEDDINGS
# ─────────────────────────────────────────────

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL)
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


# ─────────────────────────────────────────────
# CATEGORY SYSTEM
# ─────────────────────────────────────────────

RULES = [
    # ── Linux ─────────────────────────────────────────────────────────────────
    ("linux", "commands", 10, ["ls", "cd", "cp", "mv", "rm", "mkdir", "rmdir", "touch", "cat", "echo"]),
    ("linux", "shell", 10, ["bash", "zsh", "fish", "sh", "shell scripting", "terminal"]),
    ("linux", "package management", 10, ["apt", "dpkg", "dnf", "rpm", "pacman", "zypper", "apk", "snap", "flatpak"]),
    ("linux", "system administration", 10, ["systemctl", "systemd", "service", "useradd", "usermod", "passwd", "sudo"]),
    ("linux", "permissions", 10, ["chmod", "chown", "chgrp", "umask", "file permissions", "ownership"]),
    ("linux", "process management", 10, ["ps", "top", "htop", "kill", "pkill", "nice", "renice"]),
    ("linux", "networking", 10, ["ssh", "scp", "rsync", "ping", "ip", "ifconfig", "netstat", "ss"]),

    # ── Red team tools ────────────────────────────────────────────────────────
    ("tools", "nmap", 15, ["nmap", "network mapper", "port scan", "port scanning",
                           "service detection", "version detection", "nse",
                           "nmap scripting engine", "host discovery"]),
    ("tools", "hydra", 10, ["hydra", "thc hydra", "password brute force",
                            "login brute force", "credential attack",
                            "ssh brute force", "ftp brute force"]),
    ("tools", "john_the_ripper", 10, ["john the ripper", "john", "john jumbo",
                                      "password cracking", "hash cracking",
                                      "dictionary attack", "wordlist attack"]),
    ("tools", "hashcat", 10, ["hashcat", "gpu cracking", "hash cracking",
                              "password cracking", "offline cracking",
                              "dictionary attack", "mask attack"]),
    ("tools", "gobuster", 10, ["gobuster", "directory enumeration",
                               "directory brute force", "content discovery",
                               "dns enumeration", "vhost enumeration"]),
    ("tools", "ffuf", 10, ["ffuf", "fuzz faster u fool", "web fuzzing",
                           "directory fuzzing", "content discovery",
                           "parameter fuzzing"]),
    ("tools", "burp_suite", 10, ["burp", "burp suite", "burp proxy",
                                 "repeater", "intruder", "web proxy",
                                 "web application testing"]),
    ("tools", "metasploit", 10, ["metasploit", "msfconsole", "msfvenom",
                                 "meterpreter", "exploit framework",
                                 "payload generation"]),
    ("tools", "sqlmap", 10, ["sqlmap", "sql injection automation",
                             "database enumeration", "sqli exploitation",
                             "sql injection scanner"]),
    ("tools", "nikto", 10, ["nikto", "web vulnerability scanner",
                            "web server scanner", "vulnerability scanning"]),
    ("tools", "wpscan", 10, ["wpscan", "wordpress scanner",
                             "wordpress enumeration", "wordpress security"]),
    ("tools", "amass", 10, ["amass", "subdomain enumeration",
                            "asset discovery", "attack surface mapping",
                            "reconnaissance"]),
    ("tools", "subfinder", 10, ["subfinder", "subdomain enumeration",
                                "asset discovery", "reconnaissance"]),
    ("tools", "enum4linux", 10, ["enum4linux", "smb enumeration",
                                 "windows enumeration", "netbios enumeration"]),
    ("tools", "impacket", 10, ["impacket", "psexec", "wmiexec",
                               "secretsdump", "ntlm relay", "smb execution"]),
    ("tools", "responder", 10, ["responder", "llmnr", "nbns",
                                "name resolution poisoning", "credential capture"]),
    ("tools", "crackmapexec", 10, ["crackmapexec", "cme",
                                   "active directory enumeration",
                                   "smb enumeration", "lateral movement"]),
    ("tools", "netexec", 10, ["netexec", "nxc",
                              "active directory enumeration",
                              "smb enumeration", "lateral movement"]),
    ("tools", "wireshark", 10, ["wireshark", "packet analysis",
                                "packet capture", "network analysis",
                                "pcap analysis"]),
    ("tools", "tcpdump", 10, ["tcpdump", "packet capture",
                              "network sniffing", "pcap"]),
    ("tools", "aircrack_ng", 10, ["aircrack-ng", "aircrack",
                                  "wifi auditing", "wireless security",
                                  "wireless cracking", "wpa cracking"]),

    # ── Red Team / Web Vulns ──────────────────────────────────────────────────
    ("redteam", "reverse_shell", 10, ["reverse shell", "revshell", "bind shell", "shell backconnect",
                                      "tcp reverse connection", "attacker shell", "remote shell access",
                                      "netcat reverse shell", "bash reverse shell", "python reverse shell",
                                      "reverse_shell"]),
    ("redteam", "sql_injection", 10, ["sql injection", "sqli", "union select", "sql payload",
                                      "blind sqli", "error-based sqli", "time-based sqli",
                                      "sql_injection", "sqli exploitation"]),
    ("redteam", "xss", 10, ["xss", "cross-site scripting", "dom xss", "reflected xss",
                             "stored xss", "script injection", "html injection"]),
    ("redteam", "rce", 10, ["rce", "remote code execution", "command injection",
                             "os command", "shell exec", "code exec", "arbitrary command"]),
    ("redteam", "csrf", 9, ["csrf", "cross-site request forgery",
                             "state-changing request", "anti-csrf token"]),
    ("redteam", "ssrf", 10, ["ssrf", "server-side request forgery", "internal endpoint",
                              "metadata endpoint", "cloud ssrf"]),
    ("redteam", "xxe", 9, ["xxe", "xml external entity", "entity injection",
                            "xml injection", "external dtd"]),
    ("redteam", "idor", 9, ["idor", "insecure direct object", "broken access control",
                             "unauthorized object access", "object reference"]),
    ("redteam", "path_traversal", 9, ["path traversal", "directory traversal", "../",
                                       "dot dot slash", "local file inclusion", "lfi", "rfi"]),
    ("redteam", "auth_bypass", 10, ["authentication bypass", "auth bypass", "broken auth",
                                     "jwt attack", "session fixation", "credential stuffing",
                                     "privilege escalation", "privesc", "priv esc"]),
    ("redteam", "open_redirect", 7, ["open redirect", "url redirect", "redirect to external",
                                      "unvalidated redirect"]),
    ("redteam", "clickjacking", 7, ["clickjacking", "ui redressing", "frame injection",
                                     "x-frame-options"]),
    ("redteam", "deserialization", 9, ["insecure deserialization", "java deserialization",
                                        "pickle exploit", "deserialization gadget"]),
    ("redteam", "ssti", 9, ["ssti", "server-side template injection", "jinja2 injection",
                             "twig injection", "template injection"]),
    ("redteam", "race_condition", 8, ["race condition", "toctou", "time of check",
                                       "time of use", "concurrency bug"]),
    ("redteam", "business_logic", 8, ["business logic", "logic flaw", "price manipulation",
                                       "quantity bypass", "workflow abuse"]),
    ("redteam", "cve", 10, ["cve", "common vulnerabilities and exposures", "vulnerability id",
                             "security flaw", "exploit database", "known vulnerability",
                             "security patch", "software vulnerability", "exploit identifier"]),

    # ── Network ───────────────────────────────────────────────────────────────
    ("network", "recon", 7, ["port scan", "host discovery",
                              "service enumeration", "banner grab",
                              "network reconnaissance", "active recon"]),
    ("network", "packet_analysis", 7, ["wireshark", "tcpdump", "packet capture", "pcap",
                                        "traffic analysis", "protocol decode"]),
    ("network", "mitm", 9, ["mitm", "man in the middle", "arp spoofing", "ssl strip",
                             "arp poison", "traffic intercept"]),
    ("network", "dns", 7, ["dns poisoning", "dns hijack", "dns spoofing",
                            "dns tunneling", "domain hijack", "dns rebind"]),
    ("network", "vpn", 6, ["vpn", "wireguard", "openvpn", "ipsec",
                            "tunneling protocol", "split tunnel"]),
    ("network", "firewall", 7, ["firewall bypass", "packet filter", "egress filter",
                                 "ingress filter", "acl", "network policy"]),
    ("network", "ids_evasion", 8, ["ids evasion", "ips bypass", "fragmentation attack",
                                    "evasion technique", "signature bypass"]),

    # ── Malware ───────────────────────────────────────────────────────────────
    ("malware", "ransomware", 10, ["ransomware", "file encryption", "ransom demand",
                                    "decrypt files", "ransom note"]),
    ("malware", "trojan", 9, ["trojan", "remote access trojan", "rat", "backdoor",
                               "persistence mechanism"]),
    ("malware", "keylogger", 9, ["keylogger", "keystroke capture", "input capture",
                                  "keylogging", "keystroke logger"]),
    ("malware", "rootkit", 10, ["rootkit", "kernel module", "ring0", "kernel exploit",
                                 "ring zero", "lkm rootkit"]),
    ("malware", "botnet", 9, ["botnet", "c2", "command and control", "bot herder",
                               "ddos bot", "zombie host"]),
    ("malware", "worm", 9, ["worm", "self-replicating", "lateral movement",
                             "propagation", "network worm"]),
    ("malware", "spyware", 8, ["spyware", "adware", "pup", "stalkerware",
                                "monitoring software"]),

    # ── Code / Languages ──────────────────────────────────────────────────────
    ("code", "python", 7, [".py", "python", "django", "flask", "fastapi",
                            "asyncio", "pydantic"]),
    ("code", "javascript", 7, [".js", "javascript", "node.js", "nodejs", "typescript",
                                "ts", "ecmascript", "commonjs", "esm"]),
    ("code", "rust", 7, [".rs", "rust", "cargo", "borrow checker", "ownership",
                          "unsafe rust", "tokio"]),
    ("code", "go", 7, [".go", "golang", "goroutine", "go routine",
                        "go mod", "go context"]),
    ("code", "java", 7, [".java", "java", "jvm", "spring boot", "maven",
                          "gradle", "jar", "bytecode"]),
    ("code", "csharp", 7, [".cs", "c#", "csharp", ".net", "dotnet",
                            "asp.net", "nuget", "blazor"]),
    ("code", "cpp", 7, [".cpp", "c++", "cmake", "llvm", "stl",
                         "memory management", "pointer", "buffer"]),
    ("code", "shell", 8, ["bash", "shell script", ".sh", "zsh", "fish",
                           "posix shell", "sh script", "heredoc"]),
    ("code", "sql", 7, ["sql", "postgres", "postgresql", "mysql", "sqlite",
                         "tsql", "plpgsql", "stored procedure", "orm"]),
    ("code", "infra_as_code", 7, ["terraform", "pulumi", "cloudformation", "ansible",
                                   "puppet", "chef", "helm", "kubernetes yaml"]),

    # ── AI / ML ───────────────────────────────────────────────────────────────
    ("ai", "llm", 8, ["llm", "transformer", "embedding", "rag",
                       "retrieval augmented", "language model",
                       "gpt", "claude", "gemini"]),
    ("ai", "prompt_injection", 10, ["prompt injection", "jailbreak", "ignore previous",
                                     "system prompt leak", "indirect injection", "prompt hack",
                                     "prompt_injection"]),
    ("ai", "model_extraction", 9, ["model extraction", "model inversion",
                                    "membership inference", "training data leak",
                                    "model stealing"]),
    ("ai", "adversarial", 9, ["adversarial example", "adversarial attack",
                               "evasion attack", "poisoning attack", "trojan model"]),
    ("ai", "fine_tuning", 7, ["fine-tuning", "lora", "qlora", "peft", "sft",
                               "rlhf", "dpo", "instruction tuning"]),
    ("ai", "vector_db", 7, ["vector database", "vector store", "pinecone",
                             "weaviate", "chroma", "qdrant", "faiss", "ann"]),
    ("ai", "ml_pipeline", 7, ["mlflow", "kubeflow", "airflow", "feature store",
                               "data pipeline", "model registry"]),

    # ── Cryptography ──────────────────────────────────────────────────────────
    ("crypto", "asymmetric", 8, ["rsa", "elliptic curve", "ecc", "ecdsa", "ecdh",
                                  "public key", "private key", "pki"]),
    ("crypto", "symmetric", 8, ["aes", "des", "3des", "chacha20", "block cipher",
                                 "stream cipher", "key derivation", "kdf"]),
    ("crypto", "hashing", 8, ["sha256", "sha512", "md5", "bcrypt", "argon2",
                               "scrypt", "pbkdf2", "collision", "hash function"]),
    ("crypto", "tls", 8, ["tls", "ssl", "mtls", "certificate", "x.509",
                           "cipher suite", "pfs", "certificate pinning"]),
    ("crypto", "crypto_attack", 9, ["padding oracle", "cbc attack", "ecb mode",
                                     "length extension", "timing attack",
                                     "nonce reuse", "weak random"]),
    ("crypto", "zero_knowledge", 8, ["zero knowledge", "zk proof", "zkp",
                                      "zk-snark", "zk-stark", "commitment scheme"]),

    # ── Social Engineering ────────────────────────────────────────────────────
    ("social", "phishing", 9, ["phishing", "spear phishing", "whaling", "vishing",
                                "smishing", "credential harvest page", "credential harvesting",
                                "credential harvest", "harvest credentials"]),
    ("social", "osint", 7, ["osint", "open source intelligence", "doxing",
                             "footprinting", "recon-ng", "maltego", "shodan"]),
    ("social", "pretexting", 8, ["pretexting", "social engineering", "impersonation",
                                  "vishing script", "pretext call"]),
    ("social", "physical", 8, ["physical security", "tailgating", "badge cloning",
                                "rfid cloning", "lock picking", "physical pentest"]),

    # ── Cloud ─────────────────────────────────────────────────────────────────
    ("cloud", "aws", 7, ["aws", "s3", "ec2", "iam", "lambda", "cloudtrail",
                          "guardduty", "security group", "vpc"]),
    ("cloud", "azure", 7, ["azure", "microsoft azure", "azure ad", "entra",
                            "arm template", "azure devops", "blob storage"]),
    ("cloud", "gcp", 7, ["gcp", "google cloud", "gke", "cloud run",
                          "bigquery", "iam policy", "service account"]),
    ("cloud", "k8s_security", 9, ["kubernetes", "k8s", "pod security", "rbac", "etcd",
                                   "container escape", "namespace isolation"]),
    ("cloud", "iam", 12, ["iam", "identity access", "privilege escalation cloud",
                          "role assumption", "credential leak", "aws keys",
                          "privilege_escalation", "iam_privilege", "iam privesc"]),

    # ── Mobile ────────────────────────────────────────────────────────────────
    ("mobile", "android", 7, ["android", "apk", "smali", "adb", "intent",
                               "broadcast receiver", "content provider"]),
    ("mobile", "ios", 7, ["ios", "ipa", "jailbreak", "mach-o", "swift",
                           "objc", "objective-c", "codesign", "entitlement"]),
    ("mobile", "mobile_pentest", 8, ["frida", "objection", "burp mobile", "mobile pentest",
                                      "binary analysis", "runtime manipulation"]),

    # ── Data & Privacy ────────────────────────────────────────────────────────
    ("data", "exfiltration", 10, ["data exfiltration", "data leak", "data breach",
                                   "sensitive data", "pii leak", "phi leak"]),
    ("data", "forensics", 7, ["digital forensics", "disk image", "memory forensics",
                               "volatility", "timeline analysis", "artifact recovery"]),
    ("data", "privacy", 7, ["gdpr", "ccpa", "pii", "personal data", "data residency",
                             "right to erasure", "data retention"]),

    ("other", "other", 0, []),
]

# ─────────────────────────────────────────────
# Precomputed lookup: keyword → (cat, sub, base_score)
# ─────────────────────────────────────────────

_AMBIGUOUS_TOKENS = {
    "sh", "cat", "rm", "mv", "cp", "ls", "cd", "ps", "ip",
    "ss", "ts", "go", ".go", "rs", "cs", "js",
}

_PHRASE_INDEX: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
_TOKEN_INDEX:  dict[str, list[tuple[str, str, int]]] = defaultdict(list)

for _cat, _sub, _score, _kws in RULES:
    for _kw in _kws:
        if _kw in _AMBIGUOUS_TOKENS:
            continue
        if " " in _kw or len(_kw) > 5:
            _PHRASE_INDEX[_kw].append((_cat, _sub, _score))
        else:
            _TOKEN_INDEX[_kw].append((_cat, _sub, _score))

_KEYWORD_INDEX = {**_PHRASE_INDEX, **_TOKEN_INDEX}

_TOKEN_PATTERN_CACHE: dict[str, re.Pattern] = {}

def _token_re(token: str) -> re.Pattern:
    if token not in _TOKEN_PATTERN_CACHE:
        _TOKEN_PATTERN_CACHE[token] = re.compile(
            r"(?<![a-zA-Z0-9])" + re.escape(token) + r"(?![a-zA-Z0-9])",
            re.IGNORECASE,
        )
    return _TOKEN_PATTERN_CACHE[token]


def infer_category(source: str) -> Tuple[str, str]:
    s = source.lower()
    tally: dict[tuple[str, str], float] = defaultdict(float)

    for keyword, entries in _PHRASE_INDEX.items():
        if keyword in s:
            for cat, sub, score in entries:
                tally[(cat, sub)] += score * 2.0

    for keyword, entries in _TOKEN_INDEX.items():
        if _token_re(keyword).search(s):
            for cat, sub, score in entries:
                tally[(cat, sub)] += score * 1.0

    if not tally:
        return "other", "other"

    best_cat, best_sub = max(
        tally,
        key=lambda k: (tally[k], len(k[1])),
    )
    return best_cat, best_sub


# ─────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks = []
    current = ""
    for p in paragraphs:
        candidate = (current + "\n\n" + p).strip() if current else p
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                current = current[-overlap:] + "\n\n" + p
            else:
                chunks.append(p)
                current = p[-overlap:]
    if current:
        chunks.append(current)
    return chunks


# ─────────────────────────────────────────────
# UPSERT
# ─────────────────────────────────────────────

def upsert_chunks(chunks: list[str], source: str, collection=None) -> int:
    if not chunks:
        return 0

    conn = get_chroma_collection()
    cur = conn.cursor()
    embedder = get_embedder()
    category, subcategory = infer_category(source)
    embeddings = embedder.encode(chunks).tolist()

    try:
        for i, (text, vec) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO documents (source, chunk_index, text, category, subcategory, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::vector)
                """,
                (source, i, text, category, subcategory, vec),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return len(chunks)


# ─────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────

RAG_MIN_LENGTH = 10

RAG_SKIP_PHRASES = {
    "hi", "hello", "hey", "yo", "sup", "howdy",
    "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "whats up",
    "thanks", "thank you", "ok", "okay", "bye", "goodbye", "what is your name",
}


def is_retrieval_query(query: str) -> bool:
    q = query.strip().lower()

    if len(q) < RAG_MIN_LENGTH:
        return False

    if q in RAG_SKIP_PHRASES:
        return False

    for keyword in _KEYWORD_INDEX:
        if keyword in q:
            return True

    return len(q.split()) >= 5


# ── Web-fallback thresholds (edit freely) ────────────────────────────────────
WEB_FALLBACK_MIN_HITS  = 2
WEB_FALLBACK_MIN_SCORE = 0.55
WEB_FALLBACK_AVG_SCORE = 0.50
WEB_FALLBACK_MAX_PAGES = 4
WEB_FALLBACK_DELAY     = 0.8
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────
# MULTI-SOURCE WEB SEARCH
# ─────────────────────────────────────────────

_SEARCH_TIMEOUT  = 10
_PARALLEL_WORKERS = 3

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Domain trust bonuses
_DOMAIN_TRUST: dict[str, float] = {
    "github.com": 2.0,
    "gitlab.com": 1.8,
    "raw.githubusercontent.com": 1.5,
    "docs.python.org": 2.0,
    "man7.org": 1.8,
    "linux.die.net": 1.8,
    "kernel.org": 2.0,
    "owasp.org": 2.5,
    "portswigger.net": 2.5,
    "nvd.nist.gov": 2.5,
    "cve.mitre.org": 2.5,
    "exploit-db.com": 2.0,
    "hacktricks.xyz": 2.0,
    "book.hacktricks.xyz": 2.0,
    "pentestmonkey.net": 1.8,
    "gtfobins.github.io": 2.0,
    "lolbas-project.github.io": 2.0,
    "stackoverflow.com": 1.5,
    "superuser.com": 1.2,
    "askubuntu.com": 1.2,
    "debian.org": 1.8,
    "archlinux.org": 1.8,
    "redhat.com": 1.5,
    "ubuntu.com": 1.5,
    "krebsonsecurity.com": 1.5,
    "schneier.com": 1.5,
    "theregister.com": 1.2,
    "bleepingcomputer.com": 1.5,
    "securityweek.com": 1.3,
    "sans.org": 1.8,
    "cisco.com": 1.3,
    "paloaltonetworks.com": 1.3,
    "tryhackme.com": 1.5,
    "hackthebox.com": 1.5,
    "ctftime.org": 1.5,
}

# Domain penalties
_DOMAIN_PENALTY: dict[str, float] = {
    "pinterest.com": -5.0,
    "pinterest.co.uk": -5.0,
    "quora.com": -2.0,
    "medium.com": -0.5,
    "reddit.com": -0.5,
    "scribd.com": -3.0,
    "slideshare.net": -2.0,
    "chegg.com": -3.0,
    "coursehero.com": -3.0,
    "answers.yahoo.com": -4.0,
}

# Domains to skip entirely
_DOMAIN_BLOCKLIST: set[str] = {
    "facebook.com", "twitter.com", "x.com", "instagram.com",
    "tiktok.com", "youtube.com", "youtu.be",
    "amazon.com", "ebay.com", "etsy.com",
    "yelp.com", "tripadvisor.com",
    "duckduckgo.com", "bing.com", "google.com", "mojeek.com",
}

# Path quality signals
_PATH_BONUSES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"/wiki/",        re.I), 0.5),
    (re.compile(r"/docs?/",       re.I), 0.8),
    (re.compile(r"/manual/",      re.I), 0.8),
    (re.compile(r"/tutorial/",    re.I), 0.6),
    (re.compile(r"/writeup",      re.I), 0.8),
    (re.compile(r"/exploit",      re.I), 0.7),
    (re.compile(r"/vulnerabilit", re.I), 0.7),
    (re.compile(r"/cve-\d{4}-",   re.I), 1.0),
    (re.compile(r"/advisory",     re.I), 0.8),
    (re.compile(r"\.md$",         re.I), 0.5),
    (re.compile(r"\.rst$",        re.I), 0.4),
]

_PATH_PENALTIES: list[tuple[re.Pattern, float]] = [
    (re.compile(r"/tag/",             re.I), -0.5),
    (re.compile(r"/category/",        re.I), -0.5),
    (re.compile(r"/author/",          re.I), -0.8),
    (re.compile(r"/search\?",         re.I), -2.0),
    (re.compile(r"/page/\d+",         re.I), -0.3),
    (re.compile(r"\?.*utm_",          re.I), -0.2),
    (re.compile(r"login|signin",      re.I), -3.0),
    (re.compile(r"paywall|subscribe", re.I), -2.0),
]


def _fetch_html(url: str, timeout: int = _SEARCH_TIMEOUT) -> str:
    """GET a URL and return the decoded body, or '' on any error."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("_fetch_html failed for %s: %s", url, exc)
        return ""


def _extract_hrefs(html: str) -> list[str]:
    """Pull all href= URLs from raw HTML."""
    urls = []
    for m in re.finditer(r'href=["\']?(https?://[^"\'>\s]+)', html):
        u = _html.unescape(m.group(1))
        if u not in urls:
            urls.append(u)
    return urls


def _search_ddg(query: str, max_results: int) -> list[str]:
    """DuckDuckGo HTML endpoint with uddg= redirect extraction."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    for m in re.finditer(r'uddg=(https?%3A%2F%2F[^&"]+)', body):
        real = urllib.parse.unquote(m.group(1))
        if "duckduckgo.com" not in real and real not in urls:
            urls.append(real)
        if len(urls) >= max_results:
            break

    if not urls:
        for u in _extract_hrefs(body):
            if "duckduckgo.com" not in u:
                urls.append(u)
            if len(urls) >= max_results:
                break

    logger.debug("DDG returned %d URLs for %r", len(urls), query)
    return urls[:max_results]


def _search_bing(query: str, max_results: int) -> list[str]:
    """Bing HTML scrape."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.bing.com/search?q={encoded}&count={max_results * 2}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    for m in re.finditer(r'<cite[^>]*>(https?://[^<]+)</cite>', body):
        u = _html.unescape(m.group(1)).strip()
        if "bing.com" not in u and u not in urls:
            urls.append(u)
        if len(urls) >= max_results:
            break

    if not urls:
        for u in _extract_hrefs(body):
            if "bing.com" not in u and "microsoft.com" not in u:
                urls.append(u)
            if len(urls) >= max_results:
                break

    logger.debug("Bing returned %d URLs for %r", len(urls), query)
    return urls[:max_results]


def _search_mojeek(query: str, max_results: int) -> list[str]:
    """Mojeek — independent crawler index."""
    encoded = urllib.parse.quote_plus(query)
    url = f"https://www.mojeek.com/search?q={encoded}&l={max_results * 2}"
    body = _fetch_html(url)
    if not body:
        return []

    urls: list[str] = []
    for m in re.finditer(
        r'class=["\']ob["\'][^>]*href=["\']([^"\']+)["\']'
        r'|href=["\']([^"\']+)["\'][^>]*class=["\']ob["\']',
        body,
    ):
        u = _html.unescape(m.group(1) or m.group(2) or "").strip()
        if u.startswith("http") and "mojeek.com" not in u and u not in urls:
            urls.append(u)
        if len(urls) >= max_results:
            break

    if not urls:
        for u in _extract_hrefs(body):
            if "mojeek.com" not in u:
                urls.append(u)
            if len(urls) >= max_results:
                break

    logger.debug("Mojeek returned %d URLs for %r", len(urls), query)
    return urls[:max_results]


def _score_url(url: str) -> float:
    """Score a URL by domain trust, path quality, and spam signals."""
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return -99.0

    domain = parsed.netloc.lower().removeprefix("www.")
    path   = parsed.path + ("?" + parsed.query if parsed.query else "")

    if any(domain == b or domain.endswith("." + b) for b in _DOMAIN_BLOCKLIST):
        return -99.0

    score = 0.0

    for trusted, bonus in _DOMAIN_TRUST.items():
        if domain == trusted or domain.endswith("." + trusted):
            score += bonus
            break

    for penalised, penalty in _DOMAIN_PENALTY.items():
        if domain == penalised or domain.endswith("." + penalised):
            score += penalty
            break

    for pattern, bonus in _PATH_BONUSES:
        if pattern.search(path):
            score += bonus

    for pattern, penalty in _PATH_PENALTIES:
        if pattern.search(path):
            score += penalty

    if parsed.scheme != "https":
        score -= 0.5

    depth = path.count("/")
    if depth <= 3:
        score += 0.3
    elif depth >= 7:
        score -= 0.3

    return score


def _multi_search(query: str, max_results: int = 6) -> list[str]:
    """
    Query DDG, Bing, and Mojeek in parallel, merge results, score and
    deduplicate by normalised URL key, return top max_results URLs.
    """
    per_backend = max_results + 4

    backends = [
        (_search_ddg,    query, per_backend),
        (_search_bing,   query, per_backend),
        (_search_mojeek, query, per_backend),
    ]

    raw_urls: list[str] = []
    with ThreadPoolExecutor(max_workers=_PARALLEL_WORKERS) as pool:
        futures = {pool.submit(fn, q, n): fn.__name__ for fn, q, n in backends}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results = future.result()
                raw_urls.extend(results)
                logger.debug("%s contributed %d URLs", name, len(results))
            except Exception as exc:
                logger.warning("Search backend %s failed: %s", name, exc)

    if not raw_urls:
        logger.warning("_multi_search: all backends returned nothing for %r", query)
        return []

    seen_keys: set[str] = set()
    scored: list[tuple[float, str]] = []

    for url in raw_urls:
        score = _score_url(url)
        if score < -10:
            continue

        try:
            parsed = urllib.parse.urlparse(url)
            key = (
                parsed.netloc.lower().removeprefix("www.")
                + parsed.path.rstrip("/").lower()
            )
        except Exception:
            key = url

        if key in seen_keys:
            continue
        seen_keys.add(key)
        scored.append((score, url))

    scored.sort(key=lambda t: t[0], reverse=True)
    top = [url for _, url in scored[:max_results]]

    logger.info(
        "_multi_search: %d raw → %d unique → top %d for %r",
        len(raw_urls), len(scored), len(top), query,
    )
    return top


# ─────────────────────────────────────────────
# FETCH + CLEAN
# ─────────────────────────────────────────────

def _fetch_and_clean(url: str) -> str:
    """
    Fetch a URL and return cleaned plain text.
    Returns empty string on any failure so the caller silently skips bad pages.
    """
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0 Safari/537.36"
        )
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=12) as resp:
            content_type = resp.headers.get("Content-Type", "")
            if "text" not in content_type:
                return ""
            raw_html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("Fallback fetch failed for %s: %s", url, exc)
        return ""

    text = ""
    try:
        import trafilatura
        text = trafilatura.extract(raw_html) or ""
    except ImportError:
        pass

    if not text or len(text) < 200:
        try:
            soup = BeautifulSoup(raw_html, "html.parser")
            for tag in soup(["script", "style", "noscript", "nav", "footer"]):
                tag.decompose()
            main = soup.find("article") or soup.find("main") or soup.body
            text = "\n".join(main.stripped_strings) if main else ""
        except Exception:
            pass

    clean_lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or len(s) < 4:
            continue
        letters = sum(1 for c in s if c.isalpha())
        if len(s) > 0 and letters / len(s) < 0.4:
            continue
        clean_lines.append(s)

    return "\n".join(clean_lines)


def _build_signal_from_url(url: str, content: str = "") -> str:
    """
    Derive a clean categorisation signal from a URL + optional content hint.
    """
    raw = unquote(url)

    if "github.com" in raw:
        m = re.search(r"/(?:blob|tree)/[^/]+/(.+)$", raw)
        path_part = m.group(1) if m else raw.split("/")[-1]
    else:
        p_parsed = urlparse(raw)
        path_part = p_parsed.path.rstrip("/")
        if "/" in path_part:
            parts = [p for p in path_part.split("/") if p]
            path_part = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else url)

    signal = re.sub(r"[/_\-%.]+", " ", path_part).strip()

    for line in content.splitlines():
        s = line.strip().lstrip("#").strip()
        if len(s) >= 8 and re.search(r"[a-zA-Z]{3,}", s):
            signal = signal + " " + s[:120]
            break

    return signal.lower()


def _web_fallback_ingest(query: str) -> int:
    """
    Search the web for *query*, fetch the top results, chunk and ingest
    anything useful.  Returns the number of new chunks stored.
    """
    logger.info("RAG web fallback triggered for query: %r", query)
    candidate_urls = _multi_search(query, max_results=WEB_FALLBACK_MAX_PAGES + 2)
    if not candidate_urls:
        logger.warning("Web fallback: no results from any search backend")
        return 0

    total_new = 0
    ingested = 0

    for url in candidate_urls:
        if ingested >= WEB_FALLBACK_MAX_PAGES:
            break

        time.sleep(WEB_FALLBACK_DELAY)
        text = _fetch_and_clean(url)

        if not text or len(text.split()) < 40:
            logger.debug("Web fallback: skipped %s (too short/empty)", url)
            continue

        signal = _build_signal_from_url(url, text)
        chunks = chunk_text(text)

        seen: set[str] = set()
        good_chunks = []
        for c in chunks:
            norm = " ".join(c.lower().split())
            if norm not in seen and len(c.split()) >= 5:
                seen.add(norm)
                good_chunks.append(c)

        if not good_chunks:
            continue

        n = upsert_chunks(good_chunks, source=signal)
        total_new += n
        ingested += 1
        logger.info("Web fallback: +%d chunks from %s", n, url)

    logger.info("Web fallback done: %d chunks from %d pages", total_new, ingested)
    return total_new


def _raw_vector_search(
    qvec: list[float],
    top_k: int,
    min_score: float,
) -> list[dict]:
    """Run a pgvector cosine search and return scored hit dicts."""
    conn = get_chroma_collection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index,
                   embedding <-> %s::vector AS distance
            FROM documents
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, top_k),
        )
        rows = cur.fetchall()
    except Exception:
        conn.rollback()
        raise

    hits = []
    for row_id, text, source, category, subcategory, chunk_index, dist in rows:
        score = 1 / (1 + dist)
        if score >= min_score:
            hits.append({
                "id": row_id,
                "text": text,
                "source": source,
                "score": round(score, 4),
                "category": category,
                "subcategory": subcategory,
                "chunk_index": chunk_index,
            })
    return hits
def _raw_vector_search_category(
    qvec: list[float],
    category: str,
    top_k: int,
    min_score: float,
):
    conn = get_chroma_collection()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, text, source, category, subcategory, chunk_index,
               embedding <-> %s::vector AS distance
        FROM documents
        WHERE category = %s
        ORDER BY embedding <-> %s::vector
        LIMIT %s
        """,
        (qvec, category, qvec, top_k),
    )

    rows = cur.fetchall()

    hits = []
    for row_id, text, source, category, subcategory, chunk_index, dist in rows:
        score = 1 / (1 + dist)
        if score >= min_score:
            hits.append({
                "id": row_id,
                "text": text,
                "source": source,
                "score": round(score, 4),
                "category": category,
                "subcategory": subcategory,
                "chunk_index": chunk_index,
            })

    return hits

def retrieve_context(
    query: str,
    top_k: int = config.TOP_K_RESULTS,
    min_score: float = config.MIN_RELEVANCE_SCORE,
    *,
    web_fallback: bool = True,
):
    """
    Retrieve relevant chunks for *query*.

    Search order:

        1. Normal vector search (all categories)
        2. If results are weak, try category='other'
        3. If still weak, perform web fallback
        4. Re-run vector search after ingest

    Returns a list of hit dictionaries.
    """
    q = query.strip()
    if not q:
        return []

    if q.lower() in RAG_SKIP_PHRASES or len(q) < RAG_MIN_LENGTH:
        return []

    embedder = get_embedder()
    qvec = embedder.encode(query).tolist()

    # ---------------------------------------------------
    # First pass: normal vector search
    # ---------------------------------------------------
    hits = _raw_vector_search(qvec, top_k, min_score)

    # ---------------------------------------------------
    # Evaluate first-pass quality
    # ---------------------------------------------------
    avg_score = (
        sum(h["score"] for h in hits) / len(hits)
        if hits else 0.0
    )

    weak_results = (
        len(hits) < WEB_FALLBACK_MIN_HITS
        or (hits and hits[0]["score"] < WEB_FALLBACK_MIN_SCORE)
        or avg_score < WEB_FALLBACK_AVG_SCORE
    )

    # ---------------------------------------------------
    # Second pass: try "other" category before web
    # ---------------------------------------------------
    used_other = False

    if weak_results:
        conn = get_chroma_collection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id,
                   text,
                   source,
                   category,
                   subcategory,
                   chunk_index,
                   embedding <-> %s::vector AS distance
            FROM documents
            WHERE category = 'other'
            ORDER BY embedding <-> %s::vector
            LIMIT %s
            """,
            (qvec, qvec, top_k),
        )

        rows = cur.fetchall()

        other_hits = []

        for (
            row_id,
            text,
            source,
            category,
            subcategory,
            chunk_index,
            dist,
        ) in rows:
            score = 1 / (1 + dist)

            if score >= min_score:
                other_hits.append(
                    {
                        "id": row_id,
                        "text": text,
                        "source": source,
                        "score": round(score, 4),
                        "category": category,
                        "subcategory": subcategory,
                        "chunk_index": chunk_index,
                    }
                )

        if other_hits:
            logger.info(
                "Using %d hit(s) from category='other' before web fallback",
                len(other_hits),
            )
            hits = other_hits
            used_other = True

    # ---------------------------------------------------
    # Recompute scores after possible "other" search
    # ---------------------------------------------------
    avg_score = (
        sum(h["score"] for h in hits) / len(hits)
        if hits else 0.0
    )

    no_keyword_match = not is_retrieval_query(query)

    needs_fallback = (
        web_fallback
        and (
            len(hits) < WEB_FALLBACK_MIN_HITS
            or (hits and hits[0]["score"] < WEB_FALLBACK_MIN_SCORE)
            or avg_score < WEB_FALLBACK_AVG_SCORE
            or (no_keyword_match and len(q.split()) >= 4)
        )
    )

    # ---------------------------------------------------
    # Web fallback
    # ---------------------------------------------------
    used_web = False

    if needs_fallback:
        logger.info("Local retrieval weak; triggering web fallback")

        new_chunks = _web_fallback_ingest(query)

        if new_chunks > 0:
            hits = _raw_vector_search(qvec, top_k, min_score)
            used_web = True

        else:
            _get_console_print()(
                "[dim yellow]⟳ RAG web fallback triggered but found nothing useful[/]"
            )

    # ---------------------------------------------------
    # Annotate provenance
    # ---------------------------------------------------
    for h in hits:
        h["web_fallback"] = used_web
        h["other_fallback"] = used_other

    return hits


def _get_console_print():
    try:
        from rich.console import Console
        return Console().print
    except ImportError:
        return print


# ─────────────────────────────────────────────
# CONTEXT FORMATTING
# ─────────────────────────────────────────────

def format_context_block(rag_hits: list[dict]) -> str:
    if not rag_hits:
        return ""

    db_hits  = [h for h in rag_hits if not h.get("web_fallback")]
    web_hits = [h for h in rag_hits if h.get("web_fallback")]

    preamble_lines = ["The following context was retrieved to help answer the query."]

    if db_hits and web_hits:
        preamble_lines.append(
            f"{len(db_hits)} chunk(s) came from the pre-ingested local knowledge base [DB] "
            f"and {len(web_hits)} chunk(s) were fetched live from the web [WEB] because local "
            f"coverage was insufficient."
        )
        preamble_lines.append(
            "Web-sourced chunks are unvetted — treat them as helpful but potentially "
            "incomplete or inaccurate. Prefer DB chunks where they conflict."
        )
    elif web_hits:
        preamble_lines.append(
            f"All {len(web_hits)} chunk(s) were fetched live from the web [WEB] — "
            f"the local knowledge base had no relevant content for this query."
        )
        preamble_lines.append(
            "These chunks are unvetted. Cross-check claims where accuracy is critical."
        )
    else:
        preamble_lines.append(
            f"All {len(db_hits)} chunk(s) came from the pre-ingested local knowledge base [DB]."
        )

    preamble = " ".join(preamble_lines)

    blocks = []
    for hit in rag_hits:
        provenance = "WEB" if hit.get("web_fallback") else "DB"
        blocks.append(
            f"[SOURCE: {hit.get('source', 'unknown')} | "
            f"cat: {hit.get('category', 'unknown')}/{hit.get('subcategory', 'unknown')} | "
            f"chunk: {hit.get('chunk_index', '?')} | "
            f"id: {hit.get('id', '?')} | "
            f"score: {hit.get('score', '?')} | "
            f"via: {provenance}]\n"
            f"{hit.get('text', '')}"
        )

    return preamble + "\n\n" + "\n\n".join(blocks)


# ─────────────────────────────────────────────
# COUNTS / LISTING
# ─────────────────────────────────────────────

def get_db_count() -> int:
    conn = get_chroma_collection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    return cur.fetchone()[0]


def count_documents() -> int:
    return get_db_count()


def list_categories(collection=None) -> list[str]:
    conn = get_chroma_collection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT category FROM documents")
    return [r[0] for r in cur.fetchall()]


def get_by_category(category: str, subcategory: str = None, collection=None) -> list[dict]:
    conn = get_chroma_collection()
    cur = conn.cursor()

    if subcategory:
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index
            FROM documents WHERE category = %s AND subcategory = %s ORDER BY id
            """,
            (category, subcategory),
        )
    else:
        cur.execute(
            """
            SELECT id, text, source, category, subcategory, chunk_index
            FROM documents WHERE category = %s ORDER BY id
            """,
            (category,),
        )

    return [
        {"id": r[0], "text": r[1], "source": r[2],
         "category": r[3], "subcategory": r[4], "chunk_index": r[5]}
        for r in cur.fetchall()
    ]
