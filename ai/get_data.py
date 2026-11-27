"""Download the scores used for training into ./data/raw/<style>.

  python get_data.py                    Chopin (data/raw/chopin)
  python get_data.py --style romantic   the Romantic collection (data/raw/romantic): all of the Chopin
                                        sources plus original piano works by Schumann, Schubert, Liszt,
                                        Mendelssohn, Brahms, Grieg, Tchaikovsky, Rachmaninoff,
                                        Scriabin, Mussorgsky, Balakirev, Burgmüller and others

Sources (free for personal, non-commercial use; see README for licences):
  - Craig Sapp: Chopin Preludes Op. 28 (Humdrum **kern)
  - Craig Sapp: Chopin Mazurkas (Humdrum **kern)
  - ASAP dataset: Chopin score files (MusicXML) for Ballades, Etudes, Scherzos, Sonatas and more
  - Mutopia Project: 5 Nocturnes, 2 Waltzes and the Prelude Op. 45 (included in extra/mutopia, copied in)
  - Optional, with --pdmx: more Nocturnes and Waltzes from PDMX, public-domain MuseScore
    editions (downloads about 2.1 GB once; only the matching scores are kept)

Romantic sources: the Chopin sources above; the ASAP dataset's Romantic composers (MusicXML;
transcriptions such as Liszt's Paganini études and Glinka's The Lark are left out); 121 original piano
pieces from the Mutopia Project (included in extra/romantic, see CREDITS.md there); with --pdmx, the
public-domain MuseScore editions of original piano works by Romantic composers.

Run:  python get_data.py
      python get_data.py --pdmx
      python get_data.py --style romantic --pdmx
"""
import argparse
import re
import shutil
import tarfile
import csv
import io
import os
import sys
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, "data", "raw", "chopin")  # set from --style in main()
UA = {"User-Agent": "flowstave/1.0"}

RAW_BASE = "https://raw.githubusercontent.com/"
KERN = [
    ("chopin-preludes", "craigsapp/chopin-preludes/main/kern/", ['prelude28-01.krn', 'prelude28-02.krn', 'prelude28-03.krn', 'prelude28-04.krn', 'prelude28-05.krn', 'prelude28-06.krn', 'prelude28-07.krn', 'prelude28-08.krn', 'prelude28-09.krn', 'prelude28-10.krn', 'prelude28-11.krn', 'prelude28-12.krn', 'prelude28-13.krn', 'prelude28-14.krn', 'prelude28-15.krn', 'prelude28-16.krn', 'prelude28-17.krn', 'prelude28-18.krn', 'prelude28-19.krn', 'prelude28-20.krn', 'prelude28-21.krn', 'prelude28-22.krn', 'prelude28-23.krn', 'prelude28-24.krn']),
    ("chopin-mazurkas", "craigsapp/chopin-mazurkas/main/kern/", ['mazurka-50.krn', 'mazurka-51.krn', 'mazurka-52.krn', 'mazurka06-1.krn', 'mazurka06-2.krn', 'mazurka06-3.krn', 'mazurka06-4.krn', 'mazurka07-1.krn', 'mazurka07-2.krn', 'mazurka07-3.krn', 'mazurka07-4.krn', 'mazurka07-5.krn', 'mazurka17-1.krn', 'mazurka17-2.krn', 'mazurka17-3.krn', 'mazurka17-4.krn', 'mazurka24-1.krn', 'mazurka24-2.krn', 'mazurka24-3.krn', 'mazurka24-4.krn', 'mazurka30-1.krn', 'mazurka30-2.krn', 'mazurka30-3.krn', 'mazurka30-4.krn', 'mazurka33-1.krn', 'mazurka33-2.krn', 'mazurka33-3.krn', 'mazurka33-4.krn', 'mazurka41-1.krn', 'mazurka41-2.krn', 'mazurka41-3.krn', 'mazurka41-4.krn', 'mazurka50-1.krn', 'mazurka50-2.krn', 'mazurka50-3.krn', 'mazurka56-1.krn', 'mazurka56-2.krn', 'mazurka56-3.krn', 'mazurka59-1.krn', 'mazurka59-2.krn', 'mazurka59-3.krn', 'mazurka63-1.krn', 'mazurka63-2.krn', 'mazurka63-3.krn', 'mazurka67-1.krn', 'mazurka67-2.krn', 'mazurka67-3.krn', 'mazurka67-4.krn', 'mazurka68-1.krn', 'mazurka68-2.krn', 'mazurka68-3.krn', 'mazurka68-4.krn']),
]
ASAP_META = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/metadata.csv"
ASAP_RAW = "https://raw.githubusercontent.com/fosfrancesco/asap-dataset/master/"


def fetch(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:  # network hiccup: retry
            if i == tries - 1:
                raise
            print(f"  retrying ({e})")
            time.sleep(2 * (i + 1))


def get_kern(name, base, files):
    dest = os.path.join(RAW, name)
    os.makedirs(dest, exist_ok=True)
    print(f"{name}: downloading...")
    n = 0
    for fn in files:
        out = os.path.join(dest, fn)
        if not os.path.exists(out):
            try:
                data = fetch(RAW_BASE + base + fn)
            except Exception as e:
                print(f"  skipped {fn}: {e}")
                continue
            with open(out, "wb") as f:
                f.write(data)
        n += 1
    print(f"{name}: {n} files")


ROMANTIC_ASAP = ["Chopin", "Schumann", "Schubert", "Liszt", "Brahms", "Rachmaninoff", "Scriabin", "Balakirev"]
ASAP_SKIP = re.compile(r"Glinka|Paganini|_repeat/|no_repeat|no_first_repeat", re.I)  # transcriptions; duplicate versions


def get_asap(composers=("Chopin",)):
    dest = os.path.join(RAW, "asap-chopin" if list(composers) == ["Chopin"] else "asap")
    os.makedirs(dest, exist_ok=True)
    print("asap: reading the file list...")
    rows = list(csv.DictReader(io.StringIO(fetch(ASAP_META).decode("utf-8"))))
    paths = sorted({r["xml_score"] for r in rows if r.get("composer") in composers and r.get("xml_score")
                    and not ASAP_SKIP.search(r["xml_score"])})
    n = 0
    for p in paths:
        out = os.path.join(dest, p.replace("/", "__"))
        if os.path.exists(out):
            n += 1
            continue
        try:
            data = fetch(ASAP_RAW + urllib.request.quote(p))
        except Exception as e:
            print(f"  skipped {p}: {e}")
            continue
        with open(out, "wb") as f:
            f.write(data)
        n += 1
    print(f"asap: {n} scores ({', '.join(composers)})")


def get_mutopia(folder="mutopia", what="Nocturnes, Waltzes, Prelude Op. 45"):
    src = os.path.join(ROOT, "extra", folder)
    dest = os.path.join(RAW, "mutopia" if folder == "mutopia" else "mutopia-" + folder)
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in sorted(os.listdir(src)) if os.path.isdir(src) else []:
        if f.lower().endswith(".mid"):
            shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
            n += 1
    print(f"mutopia: {n} pieces ({what})")


# ---------------------------------------------------------------- PDMX (optional)
PDMX_URL = "https://zenodo.org/records/15571083/files/{}?download=1"
GENRE = re.compile(r"noct|notturn|waltz|valse|walzer|vals\b", re.I)
ROMANTIC_WHO = re.compile(r"chopin|schumann|schubert|liszt|mendelssohn|brahms|grieg|tchaikovsk|tschaikowsk|rachmanin|scriabin|"
                          r"skrjabin|john field|alkan|franck|faur[eé]|dvo[rř][aá]k|mussorgsk|moussorgsk|albeniz|alb[eé]niz|granados|"
                          r"burgm[uü]ller|stephen heller|gottschalk|smetana|balakirev|borodin|l[iy]adov|saint-sa[eë]ns|chaminade|"
                          r"moszkowski|paderewski|sibelius|glazunov|arensky|lyapunov|\breger\b|mendelsohn|hensel|wieck|\bgade\b|"
                          r"rubinstein|\bweber\b|hummel|kalkbrenner|thalberg|henselt|sgambati|martucci|chabrier", re.I)
# Not original piano music: famous orchestral, opera and song pieces that are only arranged for piano.
ROMANTIC_REJECT = re.compile(r"swan lake|nutcracker|sleeping beauty|peer gynt|mountain king|morning mood|concert|symphon|ave maria|"
                             r"st[aä]ndchen|serenade|erlk[oö]nig|forelle|trout|(?<!ohne )(?<!without )\blied(er)?\b(?! ohne)|\bsong\b(?! without)|"
                             r"hungarian dance|ungarische t[aä]nze|slavonic dance|new world|wiegenlied|cradle|lullaby|wedding march|"
                             r"hochzeitsmarsch|midsummer|overture|ouvert|opera|\baria\b|1812|marche slave|trio|quintet|carmen|"
                             r"finlandia|valse triste|danse macabre|carnival of the animals|the swan|le cygne|requiem|mass\b|"
                             r"theme from|main theme|anime|game|film|movie|piano tiles|medley|mashup|beginner|easy", re.I)
REJECT = re.compile(r"easy|simplif|beginner|arr\.|arrang|duet|4 hands|four hands|violin|cello|guitar|flute|orchestra|"
                    r"string|quartet|choir|lofi|lo-fi|remix|jazz|cover|ukulele|harp|organ|accordion|vocal|sax|clarinet|trumpet", re.I)


def download_big(url, path):
    """Download a large file with progress, resuming if a partial file exists."""
    have = os.path.getsize(path) if os.path.exists(path) else 0
    req = urllib.request.Request(url, headers=dict(UA, Range=f"bytes={have}-") if have else UA)
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        if e.code == 416:  # already complete
            return
        raise
    total = have + int(r.headers.get("Content-Length") or 0)
    mode = "ab" if have and r.status == 206 else "wb"
    done = have if mode == "ab" else 0
    last = 0
    with open(path, mode) as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                pct = f"{100 * done / total:.0f}%" if total else ""
                print(f"\r  {os.path.basename(path)}: {done / 1e6:,.0f} MB {pct}   ", end="", flush=True)
    print(f"\r  {os.path.basename(path)}: {done / 1e6:,.0f} MB, done.          ")


def get_pdmx(per_piece=2, keep=False, romantic=False):
    """Pick Chopin Nocturnes and Waltzes (or, with romantic=True, original piano works by Romantic
    composers) for piano out of PDMX and keep only those scores."""
    work = os.path.join(ROOT, "data", "pdmx")
    dest = os.path.join(RAW, "pdmx")
    os.makedirs(work, exist_ok=True)
    os.makedirs(dest, exist_ok=True)
    csv_path = os.path.join(work, "PDMX.csv")
    print("pdmx: downloading the index (225 MB)...")
    download_big(PDMX_URL.format("PDMX.csv"), csv_path)

    csv.field_size_limit(10 ** 9)
    chosen = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            g = lambda k: (r.get(k) or "").strip()
            who = " ".join(g(k) for k in ("composer_name", "artist_name", "title", "song_name", "subtitle")).lower()
            name = " ".join(g(k) for k in ("title", "song_name", "subtitle"))
            if romantic:
                m = ROMANTIC_WHO.search(who)
                if not m or REJECT.search(name + " " + g("tags")) or ROMANTIC_REJECT.search(name):
                    continue
                composer = re.sub(r"[^a-z]+", "", m.group(0).lower())[:12]
            elif "chopin" not in who or not GENRE.search(name) or REJECT.search(name + " " + g("tags")):
                continue
            else:
                composer = "chopin"
            if g("subset:no_license_conflict").lower() not in ("true", "1"):
                continue
            mxl = g("mxl")
            if not mxl or mxl.upper() in ("NA", "N/A", "NAN"):
                continue
            try:
                if int(float(g("n_tracks") or 1)) > 2:
                    continue
            except ValueError:
                pass
            # Group versions of the same piece (by opus and number when given), keep the best few.
            m = re.search(r"op(?:us)?\.?\s*(\d+)\D{0,12}?(?:no|nr|n°|num|#)\.?\s*(\d+)", name, re.I)
            key = f"op{m.group(1)}-{m.group(2)}" if m else re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()[:60]
            key = f"{composer}-{key}"
            score = (g("is_best_unique_arrangement").lower() == "true", float(g("rating") or 0), float(g("n_favorites") or 0))
            chosen.setdefault(key, []).append((score, mxl.lstrip("./"), name))
    wanted = {}
    for key, lst in chosen.items():
        for score, path, name in sorted(lst, reverse=True)[:per_piece]:
            wanted[path] = (key, name)
    print(f"pdmx: {len(wanted)} scores chosen ({len(chosen)} different pieces).")
    if not wanted:
        return

    tar_path = os.path.join(work, "mxl.tar.gz")
    print("pdmx: downloading the score archive (1.9 GB, once)...")
    download_big(PDMX_URL.format("mxl.tar.gz"), tar_path)
    print("pdmx: extracting the chosen scores...")
    n = 0
    with tarfile.open(tar_path, "r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            path = m.name.lstrip("./")
            if path not in wanted:
                continue
            key, name = wanted[path]
            safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{key}_{name}")[:80]
            with tar.extractfile(m) as src, open(os.path.join(dest, f"{safe}_{os.path.basename(path)}"), "wb") as out:
                out.write(src.read())
            n += 1
    print(f"pdmx: kept {n} scores in {dest}")
    if not keep:
        os.remove(tar_path)
        print("pdmx: removed the 1.9 GB archive (use --keep-pdmx to keep it).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="chopin", choices=["chopin", "romantic"])
    ap.add_argument("--pdmx", action="store_true", help="also fetch scores from PDMX (about 2.1 GB download): Chopin Nocturnes and Waltzes, or Romantic piano works")
    ap.add_argument("--keep-pdmx", action="store_true", help="keep the PDMX archive after extracting")
    args = ap.parse_args()
    global RAW
    RAW = os.path.join(ROOT, "data", "raw", args.style)
    os.makedirs(RAW, exist_ok=True)
    romantic = args.style == "romantic"
    for name, base, files in KERN:
        get_kern(name, base, files)
    get_asap(ROMANTIC_ASAP if romantic else ("Chopin",))
    get_mutopia()
    if romantic:
        get_mutopia("romantic", "Schumann, Schubert, Mendelssohn, Grieg, Tchaikovsky, Rachmaninoff and more")
    if args.pdmx:
        get_pdmx(per_piece=1 if romantic else 2, keep=args.keep_pdmx, romantic=romantic)
    print("\nDone. Next: python prepare_data.py" + (" --style romantic" if romantic else ""))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nDownload failed:", e)
        print("Check your internet connection and run the script again; it resumes where it stopped.")
        sys.exit(1)
