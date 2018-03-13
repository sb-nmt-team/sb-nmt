#/bin/bash
python3 transform_official_dataset.py\
	--problem hewv-en\
	--input_dir he-en/\
	--output_dir preprocessed/\
	--valid_size 0.1\
	--test_size 0.1\
	--replace_hebrew\
	--filter_english
