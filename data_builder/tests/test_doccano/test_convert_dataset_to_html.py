from doccano.convert_dataset_to_html import build_alias_table


def test_build_alias_table():
    claim = [
        {
            "content": [
                ["molecularProfile", "BRAF V600"],
                ["evidenceType", "PREDICTIVE"],
                ["significance", "resistance"],
                ["therapies", "Refametinib"],
                ["diseases", "Melanoma"],
            ],
            "meta": {
                "synonyms": [
                    {
                        "name": "Melanoma",
                        "synonyms": ["Malignant Melanoma", "Naevocarcinoma"],
                        "type": "disease",
                    },
                    {
                        "name": "Refametinib",
                        "synonyms": [
                            "BAY 869766",
                            "BAY86-9766",
                            "MEK Inhibitor RDEA119",
                            "RDEA119",
                            "Refametinib",
                        ],
                        "type": "therapy",
                    },
                    {
                        "name": "BRAF",
                        "synonyms": [
                            "B-RAF1",
                            "B-Raf proto-oncogene, serine/threonine kinase",
                            "B-raf",
                            "BRAF",
                            "BRAF-1",
                            "BRAF1",
                            "NS7",
                            "RAFB1",
                        ],
                        "type": "gene",
                    },
                    {"name": "BRAF V600", "synonyms": ["VAL600"], "type": "variant"},
                ]
            },
            "type": "table",
        }
    ]
    table_html = build_alias_table(claim)

    assert table_html != ''
    assert 'B-raf' in table_html
    assert 'Malignant' in table_html
