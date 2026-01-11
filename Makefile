.PHONY: help venv install install-hf smoke smoke-hf clean

PYTHON ?= python3
VENV_DIR ?= .venv
VENV_PY := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

help:
	@printf "%s\n" \
		"Targets:" \
		"  make venv        - Create local virtualenv in $(VENV_DIR)" \
		"  make install     - Install package (editable) into venv" \
		"  make install-hf  - Install package + HF extras into venv" \
		"  make smoke       - Run gazetteer-only smoke test" \
		"  make smoke-hf    - Run smoke test with HuggingFace model" \
		"  make clean       - Remove caches/build artifacts"

venv:
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_PIP) install -U pip

install: venv
	$(VENV_PIP) install -e .

install-hf: venv
	$(VENV_PIP) install -e ".[hf]"

smoke:
	$(VENV_DIR)/bin/location-ner \
		--csv gazetteer/csv/location_kemendagri_2025.csv \
		--fuzzy --fuzzy-threshold 90 \
		--text "Graha Suryanata berlokasi di Jl. Graha Suryanata, Jl. Raya Sumberejo No.34, Sumberejo, Kecamatan Pakal, Surabaya, Jawa Timur 60192. Perumahan ini sudah ..." \
		--jsonl

smoke-hf:
	$(VENV_DIR)/bin/location-ner \
		--csv gazetteer/csv/location_kemendagri_2025.csv \
		--fuzzy --fuzzy-threshold 90 \
		--text "1) Lokasi perumahan PRANA ESTATE — Sikumbang https://sikumbang.tapera.go.id › SKB0210052024T001 ID Lokasi: SKB0210052024T001 JAWA BARAT, KOTA SUKABUMI, Cikole, Cisarua INTI INNOVACO REAL ESTATE INDONESIA (REI). Status Rumah. Subsidi: 0 Unit.\n2) Prana Estate Sukabumi (@prana.estate) — Instagram · prana.estate 290+ pengikut Lokasi Strategis di Ciaul: Hunian kami berada di pusat pertumbuhan Kota Sukabumi dengan akses yang sangat mudah. Keamanan Legalitas: Setiap unit sudah memiliki...\n3) SITEPLAN DIGITAL : perumahan PRANA ESTATE — Sikumbang https://sikumbang.tapera.go.id › SKB0210052024T001 PRANA ESTATE. INTI INNOVACO ID Lokasi: SKB0210052024T001 Cisarua, Cikole, KOTA SUKABUMI, JAWA BARAT. KOMERSIL. 152 Kavling. 0 Pembangunan. 0 Ready Stock. 0...\n4) Prana Estate Sukabumi — Facebook · Prana Estate Sukabumi 100+ pengikut Hanya dengan BAYAR 5 JUTA, Anda bisa memiliki rumah di lokasi strategis Ciaul, Kota Sukabumi. Unit kami SIAP HUNI dengan desain modern dan fungsional...\n5) Inti Innovaco | Developer Properti Kota Bogor & Sukabumi — Inti Innovaco https://intiinnovaco.co.id Miliki hunian impian bersama Inti Innovaco. Kami hadirkan rumah premium dengan kualitas terbaik, lokasi strategis, dan legalitas terjamin. Cek proyek kami!\n6) Prana Estate kini bisa diakses dari Jl.Ciaul Pasir dan Jl ... — Instagram · prana.estate 6 suka · 4 bulan yang lalu PRANA ESTATE adalah hunian strategis di Sukabumi yang menawarkan pemandangan alam Gunung Gede Pangrango dan fasilitas umum lengkap.\n7) Prana Estate: Properti Dijual & Disewa Murah - Harga 2026 — Rumah123 https://www.rumah123.com › Venue › Sukabumi Dijual Rumah Siap Huni pusat kota Sukabumi prana estate hanya 5 menit ke RSUD Bunut aja hanya 6 menit ke pusat kota Sukabumi sekat SDIT Pasim Pr.. Rp 350.000.000\n8) Prana Estate Sukabumi on Instagram: \\\"Jangan lewatkan ... — Instagram · prana.estate 10+ suka · 6 bulan yang lalu 10 likes, 0 comments - prana. estate on July 2, 2025: \\\"Jangan lewatkan kesempatan promo bayar 5 juta, bisa punya rumah di Prana. Lokasi di...\\\"\n9) Perumahan Prana Estate Berlokasi di Kota Sukabumi. Hawa ... — Facebook · Inti Innovaco https://m.facebook.com · 12 Mei 2016 Keseruan acara Open House Day, show unit rumah tipe Re · 13 Jun 2025 · 264 tontonan; Tahun ini, PT. Inti Innovaco berusia 44 tahun. Semoga de · 3..." \
		--hf-model cahya/bert-base-indonesian-NER \
		--ner-filter --ner-filter-min-score 0.5 \
		--jsonl

clean:
	rm -rf \
		__pycache__ \
		.pytest_cache \
		.mypy_cache \
		.ruff_cache \
		dist \
		build \
		*.egg-info
