#!/bin/bash

# Directory to save downloads
OUTPUT_DIR="/mnt/files_copied_to_mn5"
mkdir -p "$OUTPUT_DIR"

start_time=$(date +%s)
echo "$(date) - Starting dataset downloads..."
echo ""

# Array of dataset titles and URLs
DATASETS=(
"c. 1510 - 1699|https://bl.iro.bl.uk/downloads/d9b50283-33cc-412f-86e5-b4fb4cfe7ce5?locale=en"
"1700 - 1799|https://bl.iro.bl.uk/downloads/a51bcbab-1f2f-4d52-a6ce-7cf87c4c5e1a?locale=en"
"1800 - 1809|https://bl.iro.bl.uk/downloads/33369d60-5b21-47e3-a64b-1c41fe6df946?locale=en"
"1810 - 1819|https://bl.iro.bl.uk/downloads/0ec56300-f126-475c-9c70-4a2345020a2b?locale=en"
"1820 - 1829|https://bl.iro.bl.uk/downloads/a49d4b15-9b34-4431-a631-0b7dbde7fe2e?locale=en"
"1830 - 1839|https://bl.iro.bl.uk/downloads/914bd048-018f-4794-baee-770d3d8e9615?locale=en"
"1840 - 1849|https://bl.iro.bl.uk/downloads/cf0842c5-666e-4089-93ae-91c80f710c4a?locale=en"
"1850 - 1859|https://bl.iro.bl.uk/downloads/a702651d-df7d-4fdb-a5f3-962bb213ed68?locale=en"
"1860 - 1869|https://bl.iro.bl.uk/downloads/af437c6d-c7fc-46a2-9f71-8f98d9219c5a?locale=en"
"1870 - 1879|https://bl.iro.bl.uk/downloads/2105fae4-cae1-4ab2-959f-0cc200d6376f?locale=en"
"1880 - 1889|https://bl.iro.bl.uk/downloads/14133986-dfc2-43a9-9040-02519e15358e?locale=en"
"1890 - 1899|https://bl.iro.bl.uk/downloads/135c0882-1e09-4d0a-8d88-a8ac41066b10?locale=en"
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
    echo "$(date) - Downloading: OCR text derived from digitised books published $TITLE in ALTO XML"
    wget -c -O "$FILENAME" "$URL"
    echo ""
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "$(date) - All downloads finished."
echo "Total time elapsed: $elapsed seconds"