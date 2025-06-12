'use client';


import { usePathname, useRouter } from 'next/navigation';
import "./globals.css";

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  const tabs = [
    { name: 'HuggingFace', path: '/huggingFace' },
    { name: 'Spacy', path: '/spacy' },
    { name: 'Review Classifier', path: '/yelpreview' },
  ];
return (
  <html lang="en">
  <body>
    <div className="tabs p-6">
      <div className="tabs-list">
        {tabs.map((tab) => {
          const isActive = pathname === tab.path;
          return (
            <button
              key={tab.path}
              onClick={() => router.push(tab.path)}
              className={`tabs-list-item ${isActive ? 'tabs-list-item--active' : ''}`}
              type="button"
            >
              {tab.name}
            </button>
          );
        })}
      </div>
    </div>
        <main className="p-6 max-w-4xl mx-auto">{children}</main>
        <footer className="text-center text-sm mt-10 text-gray-400">
          Built with ❤️ by Snehasri Thumma – <a href="https://github.com/snehasrithumma/nlp-text-analytics" className="underline">GitHub</a>
        </footer>
      </body>
    </html>
  );
}
