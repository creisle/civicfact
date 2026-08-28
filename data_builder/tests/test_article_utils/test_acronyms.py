import pytest

from article_utils.acronyms import bracket_match, mark_acronyms, sub_acronym


@pytest.mark.parametrize(
    'text,result',
    [
        ['()', True],
        ['(())', True],
        ['{}', True],
        ['{[]}', True],
        ['{[}]', False],
        ['{', False],
        ['([}])', False],
        ['{[]', False],
    ],
)
def test_bracket_match(text, result):
    assert bracket_match(text) == result


STD_DEFN_EXTRACTION_EXAMPLES = [
    [
        'recently published phase II trial with a different FGFR-selective small molecule kinase inhibitor (SMKI)',
        {'small molecule kinase inhibitor'},
    ],
    ['\t18 (62.1)\tProgressive disease (PD)', {'Progressive disease'}],
    [') or magnetic resonance imaging (MRI)', {'magnetic resonance imaging'}],
    [
        'gene fusion confirmed by NGS or fluorescence in situ hybridisation (FISH)',
        {'fluorescence in situ hybridisation'},
    ],
    [
        'difference between FGFR2 fusions compared to other aberrations (e.g. mutations or amplifications). However, progression-free survival (PFS)',
        {'progression-free survival'},
    ],
    ['compared to wild-type (WT)', {'wild-type'}],
    [
        'or extrahepatic (arising from the distal main bile duct). Intrahepatic cholangiocarcinomas (iCCA)',
        {'Intrahepatic cholangiocarcinomas'},
    ],
    ['). Treatment-related adverse events (AE)', {'adverse events'}],
    ['phosphate levels above the upper limit of normal (ULN)', {'upper limit of normal'}],
    ['Safety assessments included monitoring of adverse events (AEs)', {'adverse events'}],
    [
        'cyclophosphamide and carboplatin in combination with autologous hematopoietic stem cell transplantation (AHSCT) for',
        {'autologous hematopoietic stem cell transplantation'},
    ],
    [
        'demonstrated that high-dose chemotherapy (HDC) with autologous hematopoietic stem cell transplant (AHSCT)',
        {'high-dose chemotherapy', 'autologous hematopoietic stem cell transplant'},
    ],
    [
        'Cancer abbreviations are the following: (CRC) TCGA COAD/READ colorectal cancers; (EEC) endometrial cancer;',
        set(),
    ],
    [
        'with 5\' and 3\' contexts in colorectal (CRC) and endometrial (EEC) cancers',
        set(),
    ],
    [
        'We used sequentially sequencing method and multiple ligation-dependent probe amplification (MLPA) analysis',
        {'multiple ligation-dependent probe amplification'},
    ],
    [
        'rare concurrent description of PPV with Sturge-Weber syndrome (SWS),',
        {'Sturge-Weber syndrome'},
    ],
    ['Deoxyribonucleic acid (DNA) was extracted', {'Deoxyribonucleic acid'}],
    [
        'Quantitative reverse transcription-polymerase chain reaction (qRT-PCR) and immunohistochemistry (IHC) staining were used',
        {
            'immunohistochemistry',
            'Quantitative reverse transcription-polymerase chain reaction',
        },
    ],
    ['and the IMP metabolite hypoxanthine (HX) in Reh cells', {'hypoxanthine'}],
    ['most commonly acinic cell carcinoma (AciCC)', {'acinic cell carcinoma'}],
    [
        'overall consistent with lipofibromatosis-like neural tumor (LPF-NT). LPF-NT is rare',
        {'lipofibromatosis-like neural tumor'},
    ],
    [
        'The most common type of RCC, clear cell renal cell carcinoma (ccRCC), is',
        {'clear cell renal cell carcinoma'},
    ],
    [
        'risk of transformation to accelerated phase/blast phase (AP/BP)',
        {'accelerated phase/blast phase'},
    ],
    [
        'Globally, prostate cancer (PCa) is the second most frequently diagnosed cancer of men',
        {'prostate cancer'},
    ],
    [
        'In this study, we tested the relative sensitivity of a panel of GBM stem cell-like (GSC) lines to palbociclib',
        {'GBM stem cell-like'},
    ],
    [
        'revealed two to four GBM subtypes: proneural (PN) and mesenchymal (MES) have been most reliably established, with classical (CL) and neural subtypes also described',
        {'proneural', 'mesenchymal', 'classical'},
    ],
    [
        'Hepatocellular carcinoma (HCC) is the fifth most common type of cancers worldwide.',
        {'Hepatocellular carcinoma'},
    ],
    [
        'were known to be associated with prostate cancer (PCa) risk with conflicting results',
        {'prostate cancer'},
    ],
]


@pytest.mark.parametrize(
    'input,outputs',
    STD_DEFN_EXTRACTION_EXAMPLES,
)
def test_define_acronym(input, outputs):
    all_defns = set()
    for _, defns in mark_acronyms(input).items():
        all_defns.update(defns)
    assert outputs == all_defns


SPECIAL_CASES = [
    [
        'identified novel driver oncogene in invasive mucinous adenocarcinoma of the lung (IMA)',
        'invasive mucinous adenocarcinoma of the lung',
    ],
    [
        'cocktail of drugs referred to as CHOP (Cyclophosphamide, Hydroxyldaunorubicin, Oncovin, and Prednisone).',
        {'Cyclophosphamide, Hydroxyldaunorubicin, Oncovin, and Prednisone'},
    ],
    [
        'WT: wild type; N/A: Not available; AI: Aromatase inhibitor; SERM: Selective estrogen receptor modulator; SERD: Selective estrogen receptor degrader.',
        {
            'wild type',
            'Not available',
            'Aromatase inhibitor',
            'Selective estrogen receptor modulator',
            'Selective estrogen receptor degrader',
        },
    ],
    ['TCGA-DU-6393-01 Glioma (TCGA)   M2327I  Kinase  Baseline', set()],
    [
        'Nonsense mutations, small deletions and insertions, and splice site mutations were categorized as “NSS” mutations.',
        {'Nonsense mutations, small deletions and insertions, and splice site mutations'},
    ],
    [
        'Funding sources: Supported in part by: P50 CA140146-01 (CRA); P30 CA008748 (CRA); Kristen Ann Carr Foundation (CRA); and Cycle for Survival (CRA).',
        set(),
    ],
    [
        'Patients with the other principal subtype of NSCLC, lung squamous cell cancer (lung SCC), very rarely respond to these',
        'squamous cell cancer',
    ],
]


@pytest.mark.skip('TODO')
@pytest.mark.parametrize(
    'input,outputs',
    SPECIAL_CASES,
)
def test_define_acronym_special_cases(input, outputs):
    all_defns = set()
    for _, defns in mark_acronyms(input).items():
        all_defns.update(defns)
    assert outputs == all_defns


@pytest.mark.parametrize(
    'text,acronym,defn,result',
    [
        [
            'Among 59 IMAs, we found IMAs',
            'IMA',
            'invasive mucinous adenocarcinoma',
            'Among 59 IMAs (invasive mucinous adenocarcinoma), we found IMAs',
        ]
    ],
)
def test_sub_acronyms(text, acronym, defn, result):
    assert result == sub_acronym(text, acronym, defn)


ACRONYMS_AND_DEFNS = [
    [
        'the gene for the epidermal growth factor receptor (EGFR) are found',
        {'EGFR': 'epidermal growth factor receptor'},
    ],
    ['small-molecule kinase inhibitors gefitinib (Iressa) and erlotinib (Tarceva).', {}],
    [
        'who graded responses according to Response Evaluation Criteria in Solid Tumors (RECIST).',
        {'RECIST': 'Response Evaluation Criteria in Solid Tumors'},
    ],
    [
        'responsiveness to poly (ADP-ribose) polymerase (PARP) inhibitors',
        {'PARP': 'poly (ADP-ribose) polymerase'},
    ],
    ['advanced hepatocellular carcinoma (HCC)', {'HCC': 'hepatocellular carcinoma'}],
    [
        'into the activated B cell-like (ABC) and germinal center B cell-like (GCB) subtypes',
        {'ABC': 'activated B cell-like', 'GCB': 'germinal center B cell-like'},
    ],
    [
        'RECIST 1.1, progression-free survival (PFS), overall survival (OS),',
        {'PFS': 'progression-free survival', 'OS': 'overall survival'},
    ],
    [
        'with metastatic gastrointestinal stromal tumors (GIST). ',
        {'GIST': 'gastrointestinal stromal tumors'},
    ],
    [
        'Intrahepatic cholangiocarcinoma (ICC) is an aggressive liver bile duct',
        {'ICC': 'Intrahepatic cholangiocarcinoma'},
    ],
    ['In vivo, activation of apoptosis (TUNEL) and reduction of proliferation (Ki67) ', {}],
    [
        'Adult T cell leukemia/lymphoma (ATL) is a peripheral',
        {'ATL': 'Adult T cell leukemia/lymphoma'},
    ],
    ['oncogenic gain of function (GOF). ', {'GOF': 'gain of function'}],
    ['oncogenic gain of function (GoF). ', {'GoF': 'gain of function'}],
    [' mutants in cells with EZH2(WT) resulted', {}],
    [
        'by fluorescense in situ hybridization (FISH).',
        {'FISH': 'fluorescense in situ hybridization'},
    ],
    [
        'chronic myelogenous leukemia (aCML), myelodysplastic syndrome (MDS), B-lineage acute lymphoblastic leukemia (ALL), T-cell ALL, and chronic lymphocytic leukemia (CLL)',
        {
            'MDS': 'myelodysplastic syndrome',
            'ALL': 'acute lymphoblastic leukemia',
            'CLL': 'chronic lymphocytic leukemia',
        },
    ],
    [
        'Identification of a novel metabolic-related mutation (IDH1) in metastatic pancreatic cancer.',
        {},
    ],
    [
        'ms Barr program (WRS, MM), and the National Institute for Neurological Disorders and Stroke (PM). JHH ',
        {},
    ],
    [
        'the ELISA test sensitivity, expressed as mean minimal detectable dose (MDD), was',
        {'MDD': 'minimal detectable dose'},
    ],
    [
        'Lysates were cleared by centrifugation at 14,000 g for 10 minutes and protein concentrations of samples were determined using the BCA kit',
        {},
    ],
    [
        'compared with patients whose primary tumours carried only wild-type (WT) BRAF alleles',
        {'WT': 'wild-type'},
    ],
    ['(I) Genotype and allele frequencies for all polymorphisms, (II) UGT1A', {}],
    ['0.15                            (II) Haplotype frequencies', {}],
    [
        'Thyroid cancer cell lines harboring RET /PTC1 (TPC-1), RET M918T (MZ-CRC1) and RET C634W (TT) alterations, as well as TPC-1 xenografts, were treated with JAK inhibitor, AZD1480.',
        {},
    ],
    [
        'For example, in mucinous tumors the GOG is conducting a trial comparing the gastrointestinal (GI) regimen capecitabine and oxaliplatin',
        {'GI': 'gastrointestinal'},
    ],
    [
        'These were annotated to genes and compared to events in the Catalogue of Somatic Mutations in Cancer (COSMIC) using Oncotator',
        {'COSMIC': 'Catalogue of Somatic Mutations in Cancer'},
    ],
    [
        'Therapeutics in this class include drugs that suppress estrogen production (aromatase inhibitors, GnRH agonists) and direct inhibitors of the estrogen receptor (selective estrogen receptor modulators (SERM) or selective estrogen receptor degraders (SERD)).',
        {
            'SERM': 'selective estrogen receptor modulators',
            'SERD': 'selective estrogen receptor degraders',
        },
    ],
    [
        ' In patients, CASP8 SNP D302H was the only SNP that showed an association with worse overall (OS) (p = 0.0006; multiple testing corrected p -value, q -value = 0.049) and event-free survival (EFS)',
        {'EFS': 'event-free survival'},
    ],
    [
        'Furthermore, encouraging results were also obtained in the context of renal cell (RCC) and bladder carcinoma',
        {},
    ],
    [
        'staining sections of patient tumor (PA) and PDX tissues at P3 were shown. Pieces of patient samples (PA) or PDX tissues at each passage',
        {'PA': {'patient samples', 'patient tumor'}},
    ],
    [
        'a large panel of melanoma cell lines with wild-type (WT) EZH2 and non-transformed melanocytes (HEM) and dermal fibroblasts (HDF).',
        {'WT': 'wild-type'},
    ],
    [
        'DNA was similarly extracted from formalin-fixed paraffin embedded (FFPE) ',
        {'FFPE': 'formalin-fixed paraffin embedded'},
    ],
    [
        'Pediatric low-grade gliomas (PLGGs) are frequently associated with activating BRAF gene fusions, such as KIAA1549-BRAF, that aberrantly drive the mitogen activated protein kinase (MAPK) pathway. Although RAF inhibitors (RAFi) have been proven effective in BRAF-V600E mutant tumors',
        {
            'PLGGs': {'Pediatric low-grade gliomas'},
            'RAFi': {'RAF inhibitors'},
            'MAPK': ['mitogen activated protein kinase'],
        },
    ],
]


@pytest.mark.parametrize('text,acronyms', ACRONYMS_AND_DEFNS)
def test_mark_acronyms(text, acronyms):
    # reformat strings to sets if not already sets (for simplicity in defining parameter)
    multi_defn_acronyms = {
        a: set([defn]) if isinstance(defn, str) else defn for a, defn in acronyms.items()
    }
    parsed_acronyms = mark_acronyms(text)
    assert len(parsed_acronyms) == len(acronyms)
    for acronym, defns in multi_defn_acronyms.items():
        assert acronym in parsed_acronyms
        assert parsed_acronyms[acronym] == defns
