# Dataset Download & Metadata Guide

This repository aggregates multiple historical and literary corpora from different sources. Below is a concise guide describing how each dataset is obtained and information about the respective versions.

## 1. TCP Collections (EEBO, ECCO, Evans)

### Datasets

* EEBO TCP (Early English Books Online – Text Creation Partnership)
* ECCO TCP (Eighteenth Century Collections Online – TCP)
* Evans TCP (American Imprints – TCP)

### Download Strategy

These datasets were **manually downloaded** on **March 16, 2026** using the publically available Box URLs listed in the [Text Creation Partnership webpage](https://www.textpartnership.net/pages/faq.html#faq05). Box offers a [Python SDK](https://github.com/box/box-python-sdk), but this requires some cumbersome steps in order to create a token to be used for programmatic download.

Each dataset corresponds to a separate Box URL:

* https://app.box.com/s/jjzmnrx98dkvanipopz3nxkvymnjccht (EEBO)
* https://app.box.com/s/6jbuf443i145f97c4t56z3garu3u2j09 (ECCO)
* https://app.box.com/s/zj7pzfokxde4glrhebxsavbbxzyr3ogz (Evans)

After successful download, the raw files were transferred into the MareNostrum 5 filesystem using the command below:

```bash
rsync -avh --progress eebo_all.zip bsc204326@transfer1.bsc.es:/gpfs/projects/bsc100/textmachine-data/
rsync -avh --progress ecco_all.zip bsc204326@transfer1.bsc.es:/gpfs/projects/bsc100/textmachine-data/
rsync -avh --progress evans.zip bsc204326@transfer1.bsc.es:/gpfs/projects/bsc100/textmachine-data/
```




### Version

* EEBO (Phase I & II)
* ECCO (final public release)
* Evans (public release)



## 2. BL Microsoft Collection

### Description

A large corpus derived from British Library materials, partitioned into 12 chronological subsets. 

### Coverage

* Time span: **1510–1899**
* Split into **12 datasets**

### Download Strategy

Use the provided bash script:

```bash
bash bl_microsoft.sh
```

### Version

The corpus has been published in 2021 with the following DOI: https://doi.org/10.23636/r7w6-zy15. 

## 3. Zenodo-hosted Datasets

### Datasets

* HMD
* LwM
* Gallica
* FreEM
* ANRChapitres
* Lattice
* Roman18

### Download Strategy

All datasets are downloaded using the Python script:

```bash
python zenodo_downloader.py
```

Each dataset corresponds to a Zenodo record:

* https://zenodo.org/records/15056046 (HMD)
* https://zenodo.org/records/17425252 (LwM)
* https://zenodo.org/records/4751204 (Gallica)
* https://zenodo.org/records/6481135 (FreEM)
* https://zenodo.org/records/7446728 (ANRChapitres)
* https://zenodo.org/records/14178056 (Lattice)
* https://zenodo.org/records/10404966 (Roman18)

### Version

| Dataset      | Version | Publication date | 
| -----------  | ----------- | ----------- |
| HMD  | Version v1 | 2025-09-18 |
| LwM | Version v2 | 2025-MM-DD (month and day not available) |
| Gallica | Version v1 | 2021-04-02 |
| FreEM | Version 1.0.0 |2022-04-24 |
| ANRChapitres | Version v1.0.0 | 2022-12-16 |
| Lattice | Version v0.1.1 | 2024-11-18|
| Roman18     | Version Version v1.2.1  | 2023-12-21 |




## 4. CoNSSA Dataset

### Source

Cloned directly from GitHub:

```bash
git clone https://github.com/cligs/conssa.git
```

### Version

The version used is can be retrieved by checking out the commit with hash ID `8fccf669ed1bdbad309ff190f5afb0d6066d70a3` from the `master` branch.


