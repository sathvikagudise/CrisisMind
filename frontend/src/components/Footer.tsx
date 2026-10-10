import { FlaskConical } from "lucide-react";

export default function Footer() {
  return (
    <footer className="border-t border-border bg-white">
      <div className="mx-auto max-w-7xl px-4 py-12 sm:px-6 lg:px-8">
        <div className="grid gap-8 sm:grid-cols-2 lg:grid-cols-4">
          <div className="lg:col-span-1">
            <div className="flex items-center gap-2">
              <FlaskConical className="h-5 w-5 text-primary" />
              <span className="font-bold tracking-tight text-lg">
                Crisis<span className="text-primary">Mind</span>
              </span>
            </div>
            <p className="mt-2 text-sm text-gray-500 max-w-xs">
              AI-Powered Multimodal Disaster Assessment and Emergency Response Intelligence.
            </p>
            <div className="mt-4 inline-flex items-center gap-1.5 rounded-full border border-border bg-gray-50 px-3 py-1">
              <span className="h-1.5 w-1.5 rounded-full bg-success" />
              <span className="text-xs font-medium text-gray-600">v1.0 Final Release</span>
            </div>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-secondary">Product</h4>
            <ul className="space-y-2 text-sm text-gray-500">
              <li>Final Release v1.0</li>
              <li>Gated Fusion Architecture</li>
              <li>Explainable AI (Grad-CAM)</li>
              <li>Batch Processing</li>
            </ul>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-secondary">Resources</h4>
            <ul className="space-y-2 text-sm text-gray-500">
              <li>
                <a href="https://github.com/sathvikrio/CrisisMind" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">
                  GitHub Repository
                </a>
              </li>
              <li>
                <a href="http://localhost:8000/docs" target="_blank" rel="noopener noreferrer" className="hover:text-primary transition-colors">
                  API Reference (Swagger)
                </a>
              </li>
            </ul>
          </div>

          <div>
            <h4 className="mb-3 text-sm font-semibold text-secondary">Project Info</h4>
            <ul className="space-y-2 text-sm text-gray-500">
              <li>Final Year Major Project</li>
              <li>Architecture: Multimodal AI Platform</li>
              <li>Best F1 Macro: 53.75%</li>
              <li>All modules complete</li>
            </ul>
          </div>
        </div>

        <div className="mt-10 flex flex-col items-center justify-between gap-4 border-t border-border pt-6 sm:flex-row">
          <p className="text-xs text-gray-400">
            &copy; {new Date().getFullYear()} CrisisMind. Developed for Final Year Major Project.
          </p>
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <a href="https://github.com/sathvikrio/CrisisMind" target="_blank" rel="noopener noreferrer" className="flex items-center gap-1 hover:text-primary transition-colors">
              <svg viewBox="0 0 24 24" className="h-3.5 w-3.5" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>
              GitHub
            </a>
            <span>v1.0 Final Release</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
