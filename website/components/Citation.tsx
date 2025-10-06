import React from 'react';

const Citation: React.FC = () => {
    const citationText = `El Ghazouali, S.; Mhirit, Y.; Oukhrid, A.; Michelucci, U.; Nouira, H. FusionVision: A Comprehensive Approach of 3D Object Reconstruction and Segmentation from RGB-D Cameras Using YOLO and Fast Segment Anything. Sensors 2024, 24, 2889. https://doi.org/10.3390/s24092889`;

    return (
        <section id="citation" className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
                <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">Citation</h2>
                <p className="mt-4 text-lg text-gray-600">
                    If you use FusionVision in your research, please cite the following paper.
                </p>
            </div>
            <div className="mt-12 max-w-3xl mx-auto">
                <div className="bg-gray-50 rounded-lg p-6 border border-gray-200 shadow-sm">
                    <p className="text-gray-700 font-mono text-sm leading-relaxed">
                        {citationText}
                    </p>
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