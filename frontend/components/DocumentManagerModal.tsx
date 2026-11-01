"use client";

import { useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  Database,
  File,
  FileCode,
  FileText,
  HardDrive,
  Plus,
  RefreshCw,
  Search,
  Trash2,
  UploadCloud,
  X,
} from "lucide-react";
import { RAGApiClient } from "../lib/api";
import { DocumentItem, DocumentsListResponse } from "../lib/types";

interface DocumentManagerModalProps {
  isOpen: boolean;
  onClose: () => void;
  onDocumentsUpdated?: () => void;
}

const SUPPORTED_EXTS = [".pdf", ".docx", ".txt", ".md", ".csv", ".json"];

function formatBytes(bytes: number, decimals = 1): string {
  if (bytes === 0) return "0 B";
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
}

function formatDate(isoString?: string | null): string {
  if (!isoString) return "Pending Sync";
  try {
    const d = new Date(isoString);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoString;
  }
}

function getFileIcon(type: string) {
  const t = type.toLowerCase().replace(".", "");
  if (t === "pdf") return <FileText className="w-4 h-4 text-vermilion-DEFAULT" />;
  if (t === "docx" || t === "doc") return <FileCode className="w-4 h-4 text-cobalt-DEFAULT dark:text-cobalt-dark" />;
  if (t === "csv" || t === "json" || t === "sqlite3") return <Database className="w-4 h-4 text-emerald-DEFAULT" />;
  return <File className="w-4 h-4 text-ink-mutedLight dark:text-ink-mutedDark" />;
}

export function DocumentManagerModal({
  isOpen,
  onClose,
  onDocumentsUpdated,
}: DocumentManagerModalProps) {
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [totalChunks, setTotalChunks] = useState(0);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Upload staging state
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressMsg, setUploadProgressMsg] = useState<string | null>(null);

  // Feedback alerts
  const [alertError, setAlertError] = useState<string | null>(null);
  const [alertSuccess, setAlertSuccess] = useState<string | null>(null);

  // Deleting and syncing states
  const [deletingFilename, setDeletingFilename] = useState<string | null>(null);
  const [isSyncing, setIsSyncing] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const loadDocuments = async () => {
    setIsLoadingDocs(true);
    setAlertError(null);
    try {
      const data: DocumentsListResponse = await RAGApiClient.getDocuments();
      setDocuments(data.documents);
      setTotalChunks(data.total_chunks);
    } catch (err: any) {
      setAlertError(err.message || "Failed to load documents list.");
    } finally {
      setIsLoadingDocs(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadDocuments();
      setSelectedFiles([]);
      setAlertError(null);
      setAlertSuccess(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const validateFiles = (fileList: FileList | File[]) => {
    const valid: File[] = [];
    const rejected: string[] = [];

    Array.from(fileList).forEach((file) => {
      const ext = "." + file.name.split(".").pop()?.toLowerCase();
      if (!SUPPORTED_EXTS.includes(ext)) {
        rejected.push(`${file.name} (unsupported format)`);
        return;
      }
      if (file.size > 50 * 1024 * 1024) {
        rejected.push(`${file.name} (exceeds 50MB limit)`);
        return;
      }
      valid.push(file);
    });

    if (rejected.length > 0) {
      setAlertError(`Validation notice: ${rejected.join(", ")}`);
    } else {
      setAlertError(null);
    }

    if (valid.length > 0) {
      setSelectedFiles((prev) => {
        const existingNames = new Set(prev.map((f) => f.name));
        const newAdditions = valid.filter((f) => !existingNames.has(f.name));
        return [...prev, ...newAdditions];
      });
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateFiles(e.dataTransfer.files);
    }
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateFiles(e.target.files);
    }
  };

  const handleRemoveStagedFile = (name: string) => {
    setSelectedFiles((prev) => prev.filter((f) => f.name !== name));
  };

  const handleUploadAndIndex = async () => {
    if (selectedFiles.length === 0) return;
    setIsUploading(true);
    setAlertError(null);
    setAlertSuccess(null);
    setUploadProgressMsg("Uploading documents and executing Hybrid RAG re-indexation...");

    try {
      const res = await RAGApiClient.uploadDocuments(selectedFiles);
      setAlertSuccess(
        `Indexed successfully: ${res.uploaded_files.join(", ")} (${res.sync_stats?.chunks_added ?? 0} chunks generated).`
      );
      setSelectedFiles([]);
      await loadDocuments();
      if (onDocumentsUpdated) onDocumentsUpdated();
    } catch (err: any) {
      setAlertError(err.message || "Failed to upload and index documents.");
    } finally {
      setIsUploading(false);
      setUploadProgressMsg(null);
    }
  };

  const handleDeleteDocument = async (filename: string) => {
    if (!confirm(`Are you sure you want to remove "${filename}" from the corpus and purge its chunks?`)) {
      return;
    }
    setDeletingFilename(filename);
    setAlertError(null);
    setAlertSuccess(null);

    try {
      const res = await RAGApiClient.deleteDocument(filename);
      setAlertSuccess(`Document "${filename}" removed. Chunks purged.`);
      await loadDocuments();
      if (onDocumentsUpdated) onDocumentsUpdated();
    } catch (err: any) {
      setAlertError(err.message || `Failed to delete ${filename}`);
    } finally {
      setDeletingFilename(null);
    }
  };

  const handleSyncCorpus = async () => {
    setIsSyncing(true);
    setAlertError(null);
    setAlertSuccess(null);

    try {
      const res = await RAGApiClient.syncDocuments();
      setAlertSuccess(
        `Corpus re-synchronized. Chunks added: ${res.sync_stats?.chunks_added ?? 0}, removed: ${
          res.sync_stats?.chunks_removed ?? 0
        }.`
      );
      await loadDocuments();
      if (onDocumentsUpdated) onDocumentsUpdated();
    } catch (err: any) {
      setAlertError(err.message || "Corpus re-sync failed.");
    } finally {
      setIsSyncing(false);
    }
  };

  const filteredDocs = documents.filter((doc) =>
    doc.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto flex items-center justify-center p-4 animate-fade-in">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/60 transition-opacity" onClick={onClose} />

      {/* Modal Box */}
      <div className="relative w-full max-w-4xl max-h-[90vh] flex flex-col bg-surface-light dark:bg-surface-dark border border-border-strongLight dark:border-border-strongDark shadow-elevated z-10 font-mono text-xs">
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark">
          <div>
            <div className="flex items-center space-x-2">
              <span className="w-2 h-2 bg-cobalt-DEFAULT dark:bg-cobalt-dark" />
              <h2 className="text-sm font-bold uppercase tracking-wider text-ink-primaryLight dark:text-ink-primaryDark">
                02 // ENTERPRISE KNOWLEDGE CORPUS
              </h2>
            </div>
            <p className="text-xs text-ink-mutedLight dark:text-ink-mutedDark mt-0.5 font-sans">
              Corpus document ledger, incremental vector chunking & BM25 inverted index synchronizer.
            </p>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 border border-border-light dark:border-border-dark hover:border-ink-primaryLight text-ink-mutedLight dark:text-ink-mutedDark hover:text-ink-primaryLight focus:outline-none focus:ring-1 focus:ring-cobalt-DEFAULT dark:focus:ring-cobalt-dark transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Stats Ribbon */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 p-4 border-b border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark text-xs font-mono">
          <div>
            <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
              01 // DOCUMENTS
            </span>
            <span className="font-bold text-ink-primaryLight dark:text-ink-primaryDark text-sm">
              {documents.length} FILES
            </span>
          </div>
          <div>
            <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
              02 // VECTOR CHUNKS
            </span>
            <span className="font-bold text-cobalt-DEFAULT dark:text-cobalt-dark text-sm">
              {totalChunks.toLocaleString()} CHUNKS
            </span>
          </div>
          <div>
            <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
              03 // VDB STORAGE
            </span>
            <span className="font-bold text-ink-primaryLight dark:text-ink-primaryDark text-sm">
              CHROMADB (LOCAL)
            </span>
          </div>
          <div>
            <span className="block text-ink-mutedLight dark:text-ink-mutedDark text-xs font-medium">
              04 // SYNCHRONIZATION
            </span>
            <span className="font-bold text-emerald-DEFAULT text-sm flex items-center space-x-1">
              <CheckCircle2 className="w-3 h-3" />
              <span>ACTIVE</span>
            </span>
          </div>
        </div>

        {/* Alert Notifications */}
        {alertError && (
          <div className="mx-5 mt-4 p-3 border border-vermilion-DEFAULT bg-vermilion-subtleLight dark:bg-vermilion-subtleDark text-vermilion-DEFAULT flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="font-sans">{alertError}</span>
          </div>
        )}

        {alertSuccess && (
          <div className="mx-5 mt-4 p-3 border border-emerald-DEFAULT bg-emerald-subtleLight dark:bg-emerald-subtleDark text-emerald-DEFAULT flex items-center space-x-2">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0" />
            <span className="font-sans">{alertSuccess}</span>
          </div>
        )}

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {/* Drag & Drop Upload Zone */}
          <div>
            <div className="flex items-center justify-between mb-2 text-xs font-mono font-medium text-ink-mutedLight dark:text-ink-mutedDark">
              <span>UPLOAD NEW POLICY OR DOCUMENT</span>
              <span>MAX 50MB PER FILE</span>
            </div>

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`border-2 border-dashed p-6 text-center transition-colors cursor-pointer ${
                isDragging
                  ? "border-cobalt-DEFAULT bg-cobalt-subtleLight dark:bg-cobalt-subtleDark"
                  : "border-border-strongLight dark:border-border-strongDark bg-surface-subtleLight dark:bg-surface-subtleDark hover:border-cobalt-DEFAULT"
              }`}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.md,.csv,.json"
                className="hidden"
                onChange={handleFileInputChange}
              />

              <UploadCloud className="w-6 h-6 mx-auto mb-2 text-cobalt-DEFAULT dark:text-cobalt-dark" />
              <p className="font-bold text-ink-primaryLight dark:text-ink-primaryDark mb-1">
                Drop files here or click to browse
              </p>
              <p className="text-xs font-sans text-ink-secondaryLight dark:text-ink-secondaryDark mb-3">
                Supported formats: PDF, DOCX, TXT, MD, CSV, JSON
              </p>

              <div className="flex justify-center space-x-1.5 text-xs font-mono font-medium">
                {SUPPORTED_EXTS.map((ext) => (
                  <span
                    key={ext}
                    className="px-1.5 py-0.5 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark uppercase"
                  >
                    {ext.replace(".", "")}
                  </span>
                ))}
              </div>
            </div>

            {/* Staged Files Preview */}
            {selectedFiles.length > 0 && (
              <div className="mt-3 p-3 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-ink-primaryLight dark:text-ink-primaryDark">
                  <span>QUEUE ({selectedFiles.length} FILES STAGED)</span>
                  <button
                    onClick={() => setSelectedFiles([])}
                    className="text-vermilion-DEFAULT hover:underline"
                  >
                    CLEAR QUEUE
                  </button>
                </div>

                <div className="space-y-1.5 max-h-36 overflow-y-auto">
                  {selectedFiles.map((file) => (
                    <div
                      key={file.name}
                      className="flex items-center justify-between p-2 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark"
                    >
                      <div className="flex items-center space-x-2 truncate">
                        {getFileIcon(file.name.split(".").pop() || "")}
                        <span className="truncate font-sans font-medium">{file.name}</span>
                        <span className="text-ink-mutedLight dark:text-ink-mutedDark">
                          ({formatBytes(file.size)})
                        </span>
                      </div>
                      <button
                        onClick={() => handleRemoveStagedFile(file.name)}
                        className="text-ink-mutedLight dark:text-ink-mutedDark hover:text-vermilion-DEFAULT p-0.5"
                      >
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>

                <button
                  onClick={handleUploadAndIndex}
                  disabled={isUploading}
                  className="w-full h-8 bg-cobalt-DEFAULT hover:bg-cobalt-hover dark:bg-cobalt-dark text-white font-bold flex items-center justify-center space-x-2 transition-colors disabled:opacity-50"
                >
                  {isUploading ? (
                    <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                      <span>{uploadProgressMsg || "INDEXING IN PROGRESS..."}</span>
                    </>
                  ) : (
                    <>
                      <Plus className="w-3.5 h-3.5" />
                      <span>UPLOAD & EXECUTE HYBRID INDEX ({selectedFiles.length})</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Ingested Documents Table */}
          <div>
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div className="flex items-center space-x-2">
                <span className="font-bold text-ink-primaryLight dark:text-ink-primaryDark uppercase">
                  INGESTED CORPUS LEDGER
                </span>
                <span className="text-ink-mutedLight dark:text-ink-mutedDark">
                  ({filteredDocs.length} ENTRIES)
                </span>
              </div>

              <div className="flex items-center space-x-2">
                <div className="relative">
                  <Search className="w-3 h-3 absolute left-2 top-2.5 text-ink-mutedLight dark:text-ink-mutedDark" />
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search ledger..."
                    className="h-8 pl-7 pr-2.5 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark text-ink-primaryLight dark:text-ink-primaryDark text-xs focus:outline-none focus:border-cobalt-DEFAULT"
                  />
                </div>

                <button
                  onClick={handleSyncCorpus}
                  disabled={isSyncing}
                  className="h-8 px-3 border border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark hover:border-cobalt-DEFAULT focus:outline-none focus:ring-1 focus:ring-cobalt-DEFAULT dark:focus:ring-cobalt-dark text-ink-primaryLight dark:text-ink-primaryDark flex items-center space-x-1.5 transition-colors disabled:opacity-50"
                  title="Force re-sync of all documents in data/ with Chroma and BM25"
                >
                  <RefreshCw className={`w-3 h-3 ${isSyncing ? "animate-spin" : ""}`} />
                  <span>SYNC ALL</span>
                </button>
              </div>
            </div>

            {isLoadingDocs ? (
              <div className="p-8 text-center text-ink-mutedLight dark:text-ink-mutedDark">
                LOADING KNOWLEDGE LEDGER...
              </div>
            ) : filteredDocs.length === 0 ? (
              <div className="p-8 border border-dashed border-border-light dark:border-border-dark text-center text-ink-mutedLight dark:text-ink-mutedDark">
                NO MATCHING DOCUMENTS FOUND IN CORPUS
              </div>
            ) : (
              <div className="border border-border-light dark:border-border-dark overflow-x-auto">
                <table className="w-full text-left border-collapse">
                  <thead>
                    <tr className="border-b border-border-light dark:border-border-dark bg-surface-subtleLight dark:bg-surface-subtleDark text-xs text-ink-mutedLight dark:text-ink-mutedDark font-semibold uppercase">
                      <th className="p-2.5">DOCUMENT</th>
                      <th className="p-2.5">FORMAT</th>
                      <th className="p-2.5">SIZE</th>
                      <th className="p-2.5">CHUNKS</th>
                      <th className="p-2.5">LAST SYNCED</th>
                      <th className="p-2.5 text-right">ACTION</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredDocs.map((doc) => (
                      <tr
                        key={doc.name}
                        className="border-b border-border-light dark:border-border-dark hover:bg-surface-subtleLight dark:hover:bg-surface-subtleDark transition-colors"
                      >
                        <td className="p-2.5 font-medium text-ink-primaryLight dark:text-ink-primaryDark flex items-center space-x-2 truncate max-w-xs">
                          {getFileIcon(doc.type)}
                          <span className="truncate">{doc.name}</span>
                        </td>
                        <td className="p-2.5 uppercase text-ink-mutedLight dark:text-ink-mutedDark">
                          {doc.type.replace(".", "")}
                        </td>
                        <td className="p-2.5 text-ink-secondaryLight dark:text-ink-secondaryDark">
                          {formatBytes(doc.size_bytes)}
                        </td>
                        <td className="p-2.5 text-cobalt-DEFAULT dark:text-cobalt-dark font-semibold">
                          {doc.chunk_count}
                        </td>
                        <td className="p-2.5 text-ink-mutedLight dark:text-ink-mutedDark">
                          {formatDate(doc.last_synced_at)}
                        </td>
                        <td className="p-2.5 text-right">
                          <button
                            onClick={() => handleDeleteDocument(doc.name)}
                            disabled={deletingFilename === doc.name}
                            className="p-1 hover:text-vermilion-DEFAULT text-ink-mutedLight dark:text-ink-mutedDark transition-colors"
                            title="Delete file and purge vector chunks"
                          >
                            {deletingFilename === doc.name ? (
                              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                            ) : (
                              <Trash2 className="w-3.5 h-3.5" />
                            )}
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border-light dark:border-border-dark bg-canvas-light dark:bg-canvas-dark flex items-center justify-between text-xs font-mono text-ink-mutedLight dark:text-ink-mutedDark">
          <span>KORTEX VDB LEDGER ENGINE // LOCAL PERSISTENCE</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark hover:border-ink-primaryLight focus:outline-none focus:ring-1 focus:ring-cobalt-DEFAULT dark:focus:ring-cobalt-dark text-ink-primaryLight dark:text-ink-primaryDark transition-colors font-semibold"
          >
            DISMISS
          </button>
        </div>
      </div>
    </div>
  );
}
