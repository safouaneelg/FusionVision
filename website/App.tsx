import React from 'react';
import Header from './components/Header';
import Hero from './components/Hero';
import Pipeline from './components/Pipeline';
import Results from './components/Results';
import Citation from './components/Citation';
import Footer from './components/Footer';

const App: React.FC = () => {
    return (
        <div className="bg-white text-gray-800 antialiased">
            <Header />
            <main>
                <Hero />
                <div className="space-y-24 md:space-y-32 py-24 md:py-32">
                    <Pipeline />
                    <Results />
                    <Citation />
                </div>
            </main>
            <Footer />
        </div>
    );
};

export default App;