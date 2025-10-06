import React from 'react';

const Results: React.FC = () => {
    return (
        <section id="results" className="container mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center">
                <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl">Results & Evaluation</h2>
                <p className="mt-4 text-lg text-gray-600 max-w-2xl mx-auto">
                    FusionVision achieves high accuracy in object detection, segmentation, and clean 3D reconstruction, eliminating over 85% of background and noise points.
                </p>
            </div>

            <div className="mt-16 space-y-16">
                <div className="grid md:grid-cols-2 gap-8 items-center">
                    <div className="p-6 border border-gray-200 rounded-lg shadow-sm">
                        <h3 className="text-2xl font-bold text-cyan-500 mb-4">Enhanced Object Detection</h3>
                        <p className="text-gray-600 mb-6">
                            Custom training of the YOLO model significantly improves detection accuracy for specific objects compared to pre-trained models, correctly identifying items like the bottle, which were previously missed.
                        </p>
                        <img
                            src="https://www.mdpi.com/sensors/sensors-24-02889/article_deploy/html/images/sensors-24-02889-g007.png"
                            alt="YOLO Detection Comparison"
                            className="rounded-lg shadow-xl"
                        />
                    </div>
                    <div className="p-6 border border-gray-200 rounded-lg shadow-sm">
                         <h3 className="text-2xl font-bold text-cyan-500 mb-4">Accurate 3D Reconstruction</h3>
                        <p className="text-gray-600 mb-6">
                            Post-processing techniques like downsampling and denoising remove noise and artifacts from the raw point cloud, resulting in a clean and accurate 3D model of the detected object.
                        </p>
                        <img
                            src="https://www.mdpi.com/sensors/sensors-24-02889/article_deploy/html/images/sensors-24-02889-g010.png"
                            alt="3D Reconstruction Before and After"
                            className="rounded-lg shadow-xl"
                        />
                    </div>
                </div>

                <div className="text-center">
                    <h3 className="text-2xl font-bold text-cyan-500 mb-4">Real-time 3D Segmentation in Action</h3>
                    <p className="text-gray-600 mb-6 max-w-3xl mx-auto">
                        This demonstrates the complete pipeline in real-time, from object detection and segmentation in the 2D view to the live generation of a clean, isolated 3D point cloud.
                    </p>
                    <img
                        src="https://github.com/safouaneelg/FusionVision/raw/main/images/FusionVision_results.gif"
                        alt="FusionVision Real-time Results"
                        className="rounded-lg shadow-xl mx-auto border border-gray-200"
                    />
                </div>

                <div className="p-6 border border-gray-200 rounded-lg shadow-sm">
                    <h3 className="text-2xl font-bold text-cyan-500 mb-4 text-center">Model Training Performance</h3>
                    <p className="text-gray-600 mb-8 text-center max-w-3xl mx-auto">
                        The model training demonstrates strong convergence, with loss functions decreasing steadily and performance metrics like precision, recall, and mAP stabilizing at high values after ~200 epochs.
                    </p>
                    <img
                        src="https://www.mdpi.com/sensors/sensors-24-02889/article_deploy/html/images/sensors-24-02889-g006.png"
                        alt="YOLO Training Curves"
                        className="rounded-lg shadow-xl mx-auto"
                    />
                </div>
            </div>
        </section>
    );
};

export default Results;