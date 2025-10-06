import React, { useState } from 'react';

const Citation: React.FC = () => {
    const [activeTab, setActiveTab] = useState('PlainText');
    const [copyStatus, setCopyStatus] = useState('Copy');

    const citationFormats = {
        PlainText: `El Ghazouali, S.; Mhirit, Y.; Oukhrid, A.; Michelucci, U.; Nouira, H. FusionVision: A Comprehensive Approach of 3D Object Reconstruction and Segmentation from RGB-D Cameras Using YOLO and Fast Segment Anything. Sensors 2024, 24, 2889. https://doi.org/10.3390/s24092889`,
        BibTeX: `@Article{s24092889,
    AUTHOR = {El Ghazouali, Safouane and Mhirit, Yassine and Oukhrid, Abdessamad and Michelucci, Umberto and Nouira, Hassan},
    TITLE = {FusionVision: A Comprehensive Approach of 3D Object Reconstruction and Segmentation from RGB-D Cameras Using YOLO and Fast Segment Anything},
    JOURNAL = {Sensors},
    VOLUME = {24},
    YEAR = {2024},
    NUMBER = {9},
    ARTICLE-NUMBER = {2889},
    URL = {https://www.mdpi.com/1424-8220/24/9/2889},
    ISSN = {1424-8220},
    DOI = {10.3390/s24092889}
}`,
        RIS: `TY  - JOUR
AU  - El Ghazouali, Safouane
AU  - Mhirit, Yassine
AU  - Oukhrid, Abdessamad
AU  - Michelucci, Umberto
AU  - Nouira, Hassan
TI  - FusionVision: A Comprehensive Approach of 3D Object Reconstruction and Segmentation from RGB-D Cameras Using YOLO and Fast Segment Anything
JO  - Sensors
VL  - 24
IS  - 9
SP  - 2889
PY  - 2024
DO  - 10.3390/s24092889
UR  - https://www.mdpi.com/1424-8220/24/9/2889
ER  - `,
    };

    const handleCopy = () => {
        const textToCopy = citationFormats[activeTab as keyof typeof citationFormats];
        navigator.clipboard.writeText(textToCopy).then(() => {
            setCopyStatus('Copied!');
            setTimeout(() => setCopyStatus('Copy'), 2000);
        }, () => {
            setCopyStatus('Failed');
            setTimeout(() => setCopyStatus('Copy'), 2000);
        });
    };

    const handleTabChange = (tab: string) => {
        setActiveTab(tab);
        setCopyStatus('Copy'); // Reset copy button status when changing tabs
    }

    return (
        <section id="citation" className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
                <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">Citation</h2>
                <p className="mt-4 text-lg text-gray-600">
                    If you use FusionVision in your research, please cite the following paper.
                </p>
            </div>
            <div className="mt-12 max-w-3xl mx-auto">
                <div className="flex border-b border-gray-200">
                    {Object.keys(citationFormats).map((format) => (
                        <button
                            key={format}
                            onClick={() => handleTabChange(format)}
                            className={`-mb-px py-2 px-4 text-sm font-medium border-b-2 transition-colors duration-200 ${
                                activeTab === format
                                    ? 'border-cyan-500 text-cyan-600'
                                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                            }`}
                        >
                            {format}
                        </button>
                    ))}
                </div>
                <div className="relative bg-gray-50 rounded-b-lg rounded-tr-lg p-6 border border-gray-200 border-t-0 shadow-sm">
                    <pre className="text-gray-700 font-mono text-sm leading-relaxed whitespace-pre-wrap break-words">
                        <code>{citationFormats[activeTab as keyof typeof citationFormats]}</code>
                    </pre>
                    <button
                        onClick={handleCopy}
                        aria-label="Copy citation to clipboard"
                        className="absolute top-4 right-4 bg-white hover:bg-gray-100 text-gray-600 font-semibold py-1 px-3 border border-gray-300 rounded-md shadow-sm text-xs transition-all duration-200 flex items-center gap-1"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                        </svg>
                        {copyStatus}
                    </button>
                </div>
                <div className="mt-8 text-center">
                    <h4 className="text-xl font-bold text-cyan-500">WebApp Author</h4>
                    <p className="mt-2 text-gray-600">
                        Safouane El Ghazouali
                    </p>
                </div>
            </div>
        </section>
    );
};

export default Citation;
