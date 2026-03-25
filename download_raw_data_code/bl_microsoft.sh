#!/bin/bash

# Directory to save downloads
OUTPUT_DIR="/mnt/files_copied_to_mn5"
mkdir -p "$OUTPUT_DIR"

start_time=$(date +%s)
echo "$(date) - Starting dataset downloads..."
echo ""

# Array of dataset titles and URLs
DATASETS=(
"c. 1510 - 1699|https://bl.iro.bl.uk/downloads/61f58234-b370-422f-8591-8f98e46c2757?locale=en"
"1700 - 1799|https://bl.iro.bl.uk/downloads/78b4a8ec-395e-4383-831c-809faff85ad7?locale=en"
"1800 - 1809|https://bl.iro.bl.uk/downloads/91ae15cb-e08f-4abf-8396-e4742d9d4e37?locale=en"
"1810 - 1819|https://bl.iro.bl.uk/downloads/6d1a6e17-f28d-45b9-8f7a-a03cf3a96491?locale=en"
"1820 - 1829|https://bl.iro.bl.uk/downloads/ec764dbd-1ed4-4fc2-8668-b4df5c8ec451?locale=en"
"1830 - 1839|https://bl.iro.bl.uk/downloads/eab68022-0418-4df7-a401-78972514ed20?locale=en"
"1840 - 1849|https://bl.iro.bl.uk/downloads/d16d88b0-aa3f-4dfe-b728-c58d168d7b4d?locale=en"
"1850 - 1859|https://bl.iro.bl.uk/downloads/a6a44ea8-8d33-4880-8b17-f89c90e3d89a?locale=en"
"1860 - 1869|https://bl.iro.bl.uk/downloads/2e17f00f-52e6-4259-962c-b88ad60dec23?locale=en"
"1870 - 1879|https://bl.iro.bl.uk/downloads/899c3719-030c-4517-abd3-b28fdc85eed4?locale=en"
"1880 - 1889|https://bl.iro.bl.uk/downloads/ec3b8545-775b-47bd-885d-ce895263709e?locale=en"
"1890 - 1899|https://bl.iro.bl.uk/downloads/54ed2842-089a-439a-b751-2179b3ffba28?locale=en"
)

# Loop through each dataset
for entry in "${DATASETS[@]}"; do
    TITLE="${entry%%|*}"
    URL="${entry##*|}"

    # Clean title for filename
    SAFE_TITLE="${TITLE// /_}"     # replace spaces with underscores
    SAFE_TITLE="${SAFE_TITLE//./}" # remove periods in title
    FILENAME="$OUTPUT_DIR/OCR_text_${SAFE_TITLE}.zip"

    # Skip if file already exists
    if [ -f "$FILENAME" ]; then
        echo "$(date) - File already exists, skipping: $(basename "$FILENAME")"
        echo ""
        continue
    fi

    # Print message and download
    echo "$(date) - Downloading: OCR derived text + metadata derived from digitised books published $TITLE"
    wget -c -O "$FILENAME" "$URL"
    echo ""
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "$(date) - All downloads finished."
echo "Total time elapsed: $elapsed seconds"