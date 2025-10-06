import React from 'react';

const Footer: React.FC = () => {
    return (
        <footer className="bg-white border-t border-gray-200 mt-24">
            <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 text-center text-gray-500">
                <p>&copy; 2024 by the authors. This project page is a summary of the open access article published by MDPI.</p>
                <p className="mt-2">
                    Licensed under CC BY 4.0. View the original article in{' '}
                    <a href="https://www.mdpi.com/journal/sensors" target="_blank" rel="noopener noreferrer" className="text-cyan-500 hover:underline">
                        Sensors
                    </a>.
                </p>
            </div>
        </footer>
    );
};

export default Footer;