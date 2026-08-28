const getBaseUrl = () => {
    const pathname = window.location.pathname;
    if (pathname.includes('/downloads/civic-nli')) {
        return '/downloads/civic-nli';
    }
    return '';
}

const baseUrl = getBaseUrl()

const waitForElm = (selector) => {
    return new Promise(resolve => {
        if (document.querySelector(selector)) {
            return resolve(document.querySelector(selector));
        }

        const observer = new MutationObserver(mutations => {
            if (document.querySelector(selector)) {
                resolve(document.querySelector(selector));
                observer.disconnect();
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    });
};


const initTable = (selector, dataFile, columnDefs, dataFilter = () => true, gridOpts = {}) => {
    // fetch the row data to use and one ready provide it to the Grid via the Grid API
    waitForElm(selector)
        .then(() => fetch(dataFile))
        .then(response => response.json())
        .then(data => {
            console.log('data', data);
            // let the grid know which columns and what data to use
            const gridOptions = {
                defaultColDef: {
                    sortable: true,
                    filter: true,
                    width: 150
                },
                pagination: true,
                paginationAutoPageSize: true,
                columnDefs: columnDefs,
                rowData: data.records.filter(dataFilter),
                ...gridOpts
            };
            new agGrid.Grid(document.querySelector(selector), gridOptions);
        });
};


const renderEidLink = (params) => {
    return `<a href="https://civicdb.org/links/evidence/${params.value}" target="_blank" rel="noopener">EID${params.value}</a>`;
};

const renderPmcLink = (params) => {
    return `<a href="https://www.ncbi.nlm.nih.gov/labs/pmc/articles/${params.value}" target="_blank" rel="noopener">${params.value}</a> `;
};


const renderPubmedLink = (params) => {
    return `<a href="https://pubmed.ncbi.nlm.nih.gov/${params.value}" target="_blank" rel="noopener">PMID${params.value}</a> `;
};

const renderLink = (params) => {
    return `<a href="${params.value}" target="_blank" rel="noopener">${params.value}</a> `;
}


document.addEventListener('DOMContentLoaded', () => {
    initTable('#civic_abstracts', `${baseUrl}/data/civic.json`, [
        { field: "civic_eid", cellRenderer: renderEidLink, headerName: 'CIViC', width: 100, type: 'numericColumn' },
        { field: "pubmed_id", cellRenderer: renderPubmedLink, headerName: 'PubMed', type: 'numericColumn' },
        { field: 'evidence_direction', headerName: 'Direction' },
        { field: 'evidence_type', headerName: 'Type' },
        { field: 'clinical_significance', headerName: 'Significance' },
        { field: 'status', headerName: 'Review Status' },
        { field: 'molecularProfile'},
        { field: 'drugs' }
    ], (row) => !row.license && row.annotations < 1 && (row.submittedBy != 'IlluminaBioInfo' | row.status === 'ACCEPTED') && row.status != 'REJECTED');


    initTable('#civic_fulltext', `${baseUrl}/data/civic.json`, [
        { field: "civic_eid", cellRenderer: renderEidLink, headerName: 'CIViC', width: 100, type: 'numericColumn' },
        { field: "pmc_id", cellRenderer: renderPmcLink, headerName: 'PMC', type: 'numericColumn' },
        { field: 'evidence_direction', headerName: 'Direction' },
        { field: 'evidence_type', headerName: 'Type' },
        { field: 'clinical_significance', headerName: 'Significance' },
        { field: 'status', headerName: 'Review Status' },
        { field: 'molecularProfile'},
        { field: 'drugs' },
        {field: 'license'},
        {field: 'submittedBy'},
        {field: 'acceptedBy'}
    ], (row) => row.license && !row.license.includes('ND') && row.annotations < 1 && row.pmc_id && (row.submittedBy != 'IlluminaBioInfo' | row.status === 'ACCEPTED') && row.status != 'REJECTED');

    initTable('#civic_review', `${baseUrl}/data/civic.json`, [
        { field: "civic_eid", cellRenderer: renderEidLink, headerName: 'CIViC', type: 'numericColumn' },
        { field: "pmc_id", cellRenderer: renderPmcLink, headerName: 'PMC', type: 'numericColumn' },
        { field: 'evidence_direction', headerName: 'Direction' },
        { field: 'evidence_type', headerName: 'Type' },
        { field: 'status', headerName: 'Review Status' }
    ], (row) => row.annotations > 0 && row.status == 'SUBMITTED');

    initTable('#hyp_data', `${baseUrl}/data/nli.json`, [
        { field: "username" },
        { field: "civic_eid", cellRenderer: renderEidLink, headerName: 'CIViC', type: 'numericColumn'},
        { field: "pmc_id", cellRenderer: renderPmcLink, headerName: 'PMC', type: 'numericColumn' },
        { field: "example" },
        { field: 'status_label', headerName: 'Label' },
    ], (row) => row.license);

    initTable('#civic_errors', `${baseUrl}/data/nli.json`, [
        { field: "username" },
        { field: "civic_eid", cellRenderer: renderEidLink, headerName: 'CIViC', type: 'numericColumn' },
        { field: 'source', cellRenderer: renderLink },
        { field: 'errors', flex: 1 },
    ], (row) => row.fix === true);

    initTable('#civic_revisions', `${baseUrl}/data/civic.json`, [
        { field: "civic_eid", cellRenderer: renderEidLink, headerName: 'CIViC', width: 100, type: 'numericColumn' },
        { field: "pubmed_id", cellRenderer: renderPubmedLink, headerName: 'PubMed', type: 'numericColumn' },
        { field: 'evidence_direction', headerName: 'Direction' },
        { field: 'clinical_significance', headerName: 'Significance' },
        { field: 'status', headerName: 'Review Status' },
        { field: 'annotations'},
        {headerName: 'fieldsPending', field: 'revised_fields', flex: 1 }
    ], (row) => row.pending_revisions == true && row.annotations);
});
