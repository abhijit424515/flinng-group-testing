MAIN="/home/aak425/Music/"

create_hard_link() {
  local original_file="$1"
  local new_location="$2"

  if [ -e "$original_file" ]; then
    ln "$original_file" "$new_location"
    echo "Hard link created successfully."
  else
    echo "Original file '$original_file' does not exist. Skipping..."
  fi
}

mkdir -p raw/
mkdir -p data/genomes/
mkdir -p data/promethion/
mkdir -p data/proteomes/
mkdir -p data/url/
mkdir -p data/webspam/

create_hard_link $MAIN/raw/genomes-data.gz raw/genomes-data.gz
create_hard_link $MAIN/raw/promethion-data.gz raw/promethion-data.gz
create_hard_link $MAIN/raw/proteomes-data.gz raw/proteomes-data.gz
create_hard_link $MAIN/raw/url_combined.bz2 raw/url_combined.bz2
create_hard_link $MAIN/raw/webspam_wc_normalized_trigram.svm.xz raw/webspam_wc_normalized_trigram.svm.xz

create_hard_link $MAIN/data/genomes/data data/genomes/data
create_hard_link $MAIN/data/genomes/indices data/genomes/indices
create_hard_link $MAIN/data/promethion/data data/promethion/data
create_hard_link $MAIN/data/promethion/indices data/promethion/indices
create_hard_link $MAIN/data/proteomes/data data/proteomes/data
create_hard_link $MAIN/data/proteomes/indices data/proteomes/indices
create_hard_link $MAIN/data/url/data data/url/data
create_hard_link $MAIN/data/url/indices data/url/indices
create_hard_link $MAIN/data/webspam/data data/webspam/data
create_hard_link $MAIN/data/webspam/indices data/webspam/indices