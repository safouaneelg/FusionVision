import React from 'react';
import { GithubIcon, PaperIcon } from './icons';

const Hero: React.FC = () => {
    return (
        <div className="relative pt-24 pb-32 text-center bg-gray-50 overflow-hidden">
             <div aria-hidden="true" className="absolute inset-0 grid grid-cols-2 -space-x-52 opacity-30">
                <div className="blur-[106px] h-56 bg-gradient-to-br from-cyan-500 to-purple-500 "></div>
                <div className="blur-[106px] h-32 bg-gradient-to-r from-cyan-400 to-indigo-600"></div>
            </div>
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 relative">
                <h1 className="text-4xl md:text-6xl font-extrabold tracking-tight text-gray-900">
                    Fusion<span className="text-cyan-500">Vision</span>
                </h1>
                <p className="mt-4 max-w-3xl mx-auto text-lg md:text-xl text-gray-600">
                    A Comprehensive Approach of 3D Object Reconstruction and Segmentation from RGB-D Cameras Using YOLO and Fast Segment Anything.
                </p>
                <div className="mt-10 flex justify-center gap-4">
                    <a
                        href="https://github.com/safouaneelg/FusionVision"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-cyan-600 hover:bg-cyan-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-transform transform hover:scale-105"
                    >
                        <GithubIcon />
                        View on GitHub
                    </a>
                    <a
                        href="https://www.mdpi.com/1424-8220/24/9/2889"
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-2 px-6 py-3 border border-gray-300 text-base font-medium rounded-md text-cyan-600 bg-white hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-cyan-500 transition-transform transform hover:scale-105"
                    >
                        <PaperIcon />
                        Read the Paper
                    </a>
                </div>
            </div>
        </div>
    );
};

export default Hero;