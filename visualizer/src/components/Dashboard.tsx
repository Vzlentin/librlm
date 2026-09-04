'use client';

import { useState, useCallback } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { FileUploader } from './FileUploader';
import { LogViewer } from './LogViewer';
import { AsciiRLM } from './AsciiGlobe';
import { ThemeToggle } from './ThemeToggle';
import { parseLogFile } from '@/lib/parse-logs';
import { RLMLogFile } from '@/lib/types';
import { cn } from '@/lib/utils';

export function Dashboard() {
  const [logFiles, setLogFiles] = useState<RLMLogFile[]>([]);
  const [selectedLog, setSelectedLog] = useState<RLMLogFile | null>(null);

  const handleFileLoaded = useCallback((fileName: string, content: string) => {
    const parsed = parseLogFile(fileName, content);
    setLogFiles(prev => {
      if (prev.some(f => f.fileName === fileName)) {
        return prev.map(f => f.fileName === fileName ? parsed : f);
      }
      return [...prev, parsed];
    });
    setSelectedLog(parsed);
  }, []);

  if (selectedLog) {
    return (
      <LogViewer 
        logFile={selectedLog} 
        onBack={() => setSelectedLog(null)} 
      />
    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0 grid-pattern opacity-30 dark:opacity-15" />
      <div className="absolute top-0 left-1/3 w-[500px] h-[500px] bg-primary/5 rounded-full blur-3xl" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-primary/3 rounded-full blur-3xl" />
      
      <div className="relative z-10">
        {/* Header */}
        <header className="border-b border-border">
          <div className="max-w-7xl mx-auto px-6 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold tracking-tight">
                  <span className="text-primary">RLM</span>
                  <span className="text-muted-foreground ml-2 font-normal">Visualizer</span>
                </h1>
                <p className="text-sm text-muted-foreground mt-1">
                  Debug recursive language model execution traces
                </p>
              </div>
              <div className="flex items-center gap-4">
                <ThemeToggle />
                <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-mono">
                  <span className="flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                    READY
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid lg:grid-cols-2 gap-10">
            {/* Left Column - Upload & ASCII Art */}
            <div className="space-y-8">
              {/* Upload Section */}
              <div>
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">01</span>
                  Upload Log File
                </h2>
                <FileUploader onFileLoaded={handleFileLoaded} />
              </div>
              
              {/* ASCII Architecture Diagram */}
              <div className="hidden lg:block">
                <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                  <span className="text-primary font-mono">◈</span>
                  RLM Architecture
                </h2>
                <div className="bg-muted/50 border border-border rounded-lg p-4 overflow-x-auto">
                  <AsciiRLM />
                </div>
              </div>
            </div>

            {/* Right Column - Loaded Files */}
            <div className="space-y-8">
              {logFiles.length > 0 && (
                <div>
                  <h2 className="text-sm font-medium mb-3 flex items-center gap-2 text-muted-foreground">
                    <span className="text-primary font-mono">02</span>
                    Loaded Files
                  </h2>
                  <ScrollArea className="h-[200px]">
                    <div className="space-y-2 pr-4">
                      {logFiles.map((log) => (
                        <Card
                          key={log.fileName}
                          className={cn(
                            'cursor-pointer transition-all hover:scale-[1.01]',
                            'hover:border-primary/50 hover:bg-primary/5'
                          )}
                          onClick={() => setSelectedLog(log)}
                        >
                          <CardContent className="p-3">
                            <div className="flex items-center gap-3">
                              <div className="relative flex-shrink-0">
                                <div className={cn(
                                  'w-2.5 h-2.5 rounded-full',
                                  log.metadata.final.has_final
                                    ? 'bg-primary'
                                    : 'bg-muted-foreground/30'
                                )} />
                                {log.metadata.final.has_final && (
                                  <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-primary animate-ping opacity-50" />
                                )}
                              </div>
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <span className="font-mono text-xs truncate text-foreground/80">
                                    {log.fileName}
                                  </span>
                                  <Badge variant="outline" className="text-[9px] px-1.5 py-0 h-4">
                                    {log.metadata.totalIterations} iter
                                  </Badge>
                                </div>
                                <p className="text-[11px] text-muted-foreground truncate">
                                  {log.metadata.contextQuestion}
                                </p>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  </ScrollArea>
                </div>
              )}
            </div>
          </div>
        </main>

        {/* Footer */}
        <footer className="border-t border-border mt-8">
          <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
            <p className="text-[10px] text-muted-foreground font-mono">
              RLM Visualizer • Recursive Language Models
            </p>
            <p className="text-[10px] text-muted-foreground font-mono">
              Prompt → [LM ↔ REPL] → Answer
            </p>
          </div>
        </footer>
      </div>
    </div>
  );
}
