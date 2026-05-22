import { Component, HostBinding, OnInit, ViewChild, ElementRef } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { DomSanitizer, SafeUrl } from '@angular/platform-browser';
import { Subscription } from 'rxjs';
import { InvestigationService, PipelineState, InvestigationResponse } from '../../services/investigation.service';

@Component({
  selector: 'app-dashboard',
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss'],
  standalone: false
})
export class DashboardComponent implements OnInit {
  investigationForm!: FormGroup;
  selectedFile: File | null = null;
  selectedVideoUrl: SafeUrl | null = null;
  dragOver = false;
  
  isDarkTheme = false; // Default to modern Light Theme
  currentThemeClass = 'theme-azure-blue'; // Default theme palette
  currentState: PipelineState | null = null;
  isInvestigating = false;
  results: InvestigationResponse | null = null;

  // Toggle log console
  showLogConsole = true;

  // Track bookmarked frames
  bookmarks: Array<{ timestamp: string; seconds: number; query: string; summary: string }> = [];

  themes = [
    { name: 'Azure & Blue', class: 'theme-azure-blue', colors: { primary: '#3b82f6', accent: '#2563eb' }, bgLight: '#f1f5f9', bgDark: '#0f172a' },
    { name: 'Rose & Red', class: 'theme-rose-red', colors: { primary: '#f43f5e', accent: '#e11d48' }, bgLight: '#fff1f2', bgDark: '#1e1113' },
    { name: 'Magenta & Violet', class: 'theme-magenta-violet', colors: { primary: '#d946ef', accent: '#8b5cf6' }, bgLight: '#fae8ff', bgDark: '#190d24' },
    { name: 'Cyan & Orange', class: 'theme-cyan-orange', colors: { primary: '#06b6d4', accent: '#f97316' }, bgLight: '#ecfeff', bgDark: '#081d24' }
  ];

  // Expert Settings variables
  tauLow = 0.22;
  tauHigh = 0.29;
  freeOnly = false;
  fps = 1.0;

  private investigationSub: Subscription | null = null;

  // Logs list for bottom console panel
  logs: Array<{ time: string; tag: string; message: string; type: string }> = [];

  // Resizable panel height (defaults to 2/12th of viewport height)
  panelHeight = 150;
  private isResizing = false;

  @ViewChild('videoPlayer') videoPlayer!: ElementRef<HTMLVideoElement>;
  @ViewChild('logContainer') private logContainer!: ElementRef;

  @HostBinding('class') get hostClasses() {
    return `${this.currentThemeClass} ${this.isDarkTheme ? 'dark-theme' : 'light-theme'}`;
  }

  constructor(
    private fb: FormBuilder,
    private investigationService: InvestigationService,
    private sanitizer: DomSanitizer
  ) {}

  ngOnInit(): void {
    this.investigationForm = this.fb.group({
      query: ['', [Validators.required, Validators.minLength(3)]]
    });
    
    // Detect system dark/light theme preference
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    this.isDarkTheme = mediaQuery.matches;
    
    // Listen for system theme changes
    mediaQuery.addEventListener('change', (e) => {
      this.isDarkTheme = e.matches;
      this.applyDarkLightTheme();
    });

    // Apply default theme Azure & Blue
    const defaultTheme = this.themes.find(t => t.class === this.currentThemeClass) || this.themes[0];
    this.selectTheme(defaultTheme);

    // Default height for split panel (2/12th of window height)
    this.panelHeight = Math.max(120, window.innerHeight * (2 / 12));
    
    // Initialize system log
    this.addLog('SYSTEM', 'Dual-Agent Video Investigator interface loaded.', 'info');
  }

  selectTheme(theme: any): void {
    this.themes.forEach(t => {
      document.body.classList.remove(t.class);
    });
    this.currentThemeClass = theme.class;
    document.body.classList.add(theme.class);
    this.applyDarkLightTheme();
  }

  toggleLogConsole(): void {
    this.showLogConsole = !this.showLogConsole;
    this.addLog('SYSTEM', `Diagnostics console ${this.showLogConsole ? 'opened' : 'hidden'}.`, 'info');
  }

  private applyDarkLightTheme(): void {
    if (this.isDarkTheme) {
      document.body.classList.add('dark-theme');
      document.body.classList.remove('light-theme');
    } else {
      document.body.classList.add('light-theme');
      document.body.classList.remove('dark-theme');
    }
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      const file = input.files[0];
      this.selectedFile = file;
      this.selectedVideoUrl = this.sanitizer.bypassSecurityTrustUrl(URL.createObjectURL(file));
      this.bookmarks = [];
      this.addLog('INGEST', `Video file loaded: "${file.name}" (${(file.size / (1024 * 1024)).toFixed(2)} MB)`, 'success');
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragOver = true;
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragOver = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    event.stopPropagation();
    this.dragOver = false;
    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      const file = event.dataTransfer.files[0];
      if (file.type.startsWith('video/')) {
        this.selectedFile = file;
        this.selectedVideoUrl = this.sanitizer.bypassSecurityTrustUrl(URL.createObjectURL(file));
        this.bookmarks = [];
        this.addLog('INGEST', `Video file dropped: "${file.name}" (${(file.size / (1024 * 1024)).toFixed(2)} MB)`, 'success');
      }
    }
  }

  clearFile(): void {
    if (this.selectedFile) {
      this.addLog('INGEST', `Cleared video file: "${this.selectedFile.name}"`, 'info');
    }
    this.selectedFile = null;
    this.selectedVideoUrl = null;
    this.bookmarks = [];
  }

  // Seek video player to a timestamp (HH:MM:SS format)
  seekToTimestamp(timestamp: string): void {
    if (this.videoPlayer && this.videoPlayer.nativeElement) {
      const seconds = this.parseTimestampToSeconds(timestamp);
      this.videoPlayer.nativeElement.currentTime = seconds;
      this.videoPlayer.nativeElement.play().catch(err => {
        console.log('Play initiated, waiting for interaction:', err);
      });
    }
  }

  // Bookmark current playing video frame
  bookmarkCurrentTime(): void {
    if (this.videoPlayer && this.videoPlayer.nativeElement && this.selectedFile) {
      const seconds = this.videoPlayer.nativeElement.currentTime;
      const formattedTimestamp = this.formatSecondsToTimestamp(seconds);

      // Check if already bookmarked near this second
      if (this.bookmarks.some(b => Math.abs(b.seconds - seconds) < 0.5)) {
        return;
      }

      const queryText = this.investigationForm.value.query || 'Manual Bookmark';

      this.bookmarks.push({
        timestamp: formattedTimestamp,
        seconds: seconds,
        query: queryText,
        summary: `User bookmarked frame at ${formattedTimestamp}`
      });

      this.bookmarks.sort((a, b) => a.seconds - b.seconds);
    }
  }

  // Add search result frame to bookmarks list
  addBookmarkFromResult(result: any): void {
    const seconds = this.parseTimestampToSeconds(result.timestamp);
    if (!this.bookmarks.some(b => Math.abs(b.seconds - seconds) < 0.5)) {
      this.bookmarks.push({
        timestamp: result.timestamp,
        seconds: seconds,
        query: this.investigationForm.value.query || 'Search Result',
        summary: result.summary
      });
      this.bookmarks.sort((a, b) => a.seconds - b.seconds);
    }
  }

  removeBookmark(index: number): void {
    this.bookmarks.splice(index, 1);
  }

  private parseTimestampToSeconds(timestamp: string): number {
    const parts = timestamp.split(':').map(Number);
    if (parts.length === 3) {
      return parts[0] * 3600 + parts[1] * 60 + parts[2];
    } else if (parts.length === 2) {
      return parts[0] * 60 + parts[1];
    }
    return 0;
  }

  private formatSecondsToTimestamp(seconds: number): string {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }

  onTauLowChange(val: number | null): void {
    if (val !== null) {
      this.tauLow = parseFloat(val.toFixed(2));
      if (this.tauLow > this.tauHigh) {
        this.tauHigh = this.tauLow;
      }
      this.addLog('EXPERT', `Confidence threshold tau_low updated to ${this.tauLow}`, 'info');
    }
  }

  onTauHighChange(val: number | null): void {
    if (val !== null) {
      this.tauHigh = parseFloat(val.toFixed(2));
      if (this.tauHigh < this.tauLow) {
        this.tauLow = this.tauHigh;
      }
      this.addLog('EXPERT', `Confidence threshold tau_high updated to ${this.tauHigh}`, 'info');
    }
  }

  resetExpertSettings(): void {
    this.tauLow = 0.22;
    this.tauHigh = 0.29;
    this.freeOnly = false;
    this.fps = 1.0;
    this.addLog('EXPERT', `Expert settings reset to defaults (tau_low = 0.22, tau_high = 0.29, Free Only = false, FPS = 1.0)`, 'info');
  }

  cancelInvestigation(): void {
    if (this.investigationSub) {
      this.investigationSub.unsubscribe();
      this.investigationSub = null;
    }
    this.isInvestigating = false;
    this.currentState = null;
    this.addLog('SYSTEM', 'Investigation pipeline run cancelled by user.', 'warning');
  }

  onFpsSliderChange(val: number | null): void {
    if (val !== null) {
      this.fps = parseFloat(val.toFixed(1));
      this.addLog('EXPERT', `Frame extraction rate (FPS) set to ${this.fps}`, 'info');
    }
  }

  onFpsInputChange(event: Event): void {
    const target = event.target as HTMLInputElement;
    if (target && target.value) {
      let numVal = parseFloat(target.value);
      if (isNaN(numVal)) return;
      if (numVal < 0.1) numVal = 0.1;
      if (numVal > 5.0) numVal = 5.0;
      this.fps = parseFloat(numVal.toFixed(1));
      this.addLog('EXPERT', `Frame extraction rate (FPS) set to ${this.fps}`, 'info');
    }
  }

  get currentStepIndex(): number {
    if (!this.currentState) return 0;
    if (this.currentState.stage === 4) return 3;
    return this.currentState.stage - 1;
  }

  startResize(event: MouseEvent): void {
    event.preventDefault();
    this.isResizing = true;
    
    const startY = event.clientY;
    const startHeight = this.panelHeight;
    
    const mouseMoveListener = (moveEvent: MouseEvent) => {
      if (!this.isResizing) return;
      const deltaY = moveEvent.clientY - startY;
      const newHeight = startHeight - deltaY;
      
      const maxHeight = window.innerHeight * 0.8;
      this.panelHeight = Math.min(maxHeight, Math.max(100, newHeight));
    };
    
    const mouseUpListener = () => {
      this.isResizing = false;
      window.removeEventListener('mousemove', mouseMoveListener);
      window.removeEventListener('mouseup', mouseUpListener);
    };
    
    window.addEventListener('mousemove', mouseMoveListener);
    window.addEventListener('mouseup', mouseUpListener);
  }

  addLog(tag: string, message: string, type: string = 'info'): void {
    const now = new Date();
    const time = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
    this.logs.push({ time, tag, message, type });
    
    setTimeout(() => {
      this.scrollToBottom();
    }, 50);
  }

  private scrollToBottom(): void {
    if (this.logContainer && this.logContainer.nativeElement) {
      try {
        this.logContainer.nativeElement.scrollTop = this.logContainer.nativeElement.scrollHeight;
      } catch (err) {
        // Ignore scroll errors
      }
    }
  }

  startInvestigation(): void {
    if (this.investigationForm.invalid || !this.selectedFile) return;

    this.isInvestigating = true;
    this.results = null;
    this.currentState = null;
    this.logs = []; // Clear diagnostics console on new run

    const query = this.investigationForm.value.query;
    
    this.addLog('SYSTEM', `Initiating semantic video retrieval for query: "${query}"`, 'info');

    const printedSimLogs = new Set<string>();
    
    this.investigationSub = this.investigationService.investigate(query, this.selectedFile, this.tauLow, this.tauHigh, this.freeOnly, this.fps).subscribe({
      next: (state: PipelineState) => {
        const lastMsg = this.currentState?.statusMessage;
        this.currentState = state;
        
        // Stage-based real-time detailed log simulation
        if (state.stage === 1) {
          const prog = state.progress;
          if (prog >= 10 && !printedSimLogs.has('ingest_start')) {
            printedSimLogs.add('ingest_start');
            this.addLog('INGEST', 'Saving video file to uploaded storage...', 'info');
          }
          if (prog >= 35 && !printedSimLogs.has('ingest_success')) {
            printedSimLogs.add('ingest_success');
            this.addLog('INGEST', 'Video file saved to local uploads directory.', 'success');
          }
          if (prog >= 55 && !printedSimLogs.has('edge_check')) {
            printedSimLogs.add('edge_check');
            this.addLog('EDGE', 'Checking vector index cache for video...', 'info');
          }
          if (prog >= 75 && !printedSimLogs.has('edge_load')) {
            printedSimLogs.add('edge_load');
            this.addLog('EDGE', 'Loading pre-computed CLIP FAISS index from disk...', 'info');
            this.addLog('EDGE', 'Successfully loaded FAISS index.', 'success');
          }
          if (prog >= 90 && !printedSimLogs.has('edge_embed')) {
            printedSimLogs.add('edge_embed');
            this.addLog('EDGE', 'Computing text embeddings for search query...', 'info');
          }
          if (prog >= 100 && !printedSimLogs.has('edge_search')) {
            printedSimLogs.add('edge_search');
            this.addLog('EDGE', 'Searching FAISS index for candidate matches...', 'info');
          }
        } else if (state.stage === 2) {
          if (!printedSimLogs.has('router_start')) {
            printedSimLogs.add('router_start');
            this.addLog('ROUTER', `Evaluating candidates using thresholds: tau_low = ${this.tauLow}, tau_high = ${this.tauHigh}...`, 'info');
          }
          if (state.statusMessage && state.statusMessage !== lastMsg) {
            let logType = 'info';
            if (state.statusMessage.includes('ESCALATING')) {
              logType = 'warning';
            } else if (state.statusMessage.includes('Skipping')) {
              logType = 'success';
            }
            this.addLog('ROUTER', state.statusMessage, logType);
          }
        } else if (state.stage === 3) {
          if (!printedSimLogs.has('cloud_start') && state.statusMessage.includes('Escalating')) {
            printedSimLogs.add('cloud_start');
            this.addLog('CLOUD', 'Escalating candidates to Claude Haiku 4.5...', 'info');
          }
          if (!printedSimLogs.has('cloud_processing') && state.statusMessage.includes('validating')) {
            printedSimLogs.add('cloud_processing');
            this.addLog('CLOUD', 'Claude validating semantic truth and generating forensic summaries...', 'info');
          }
        }

        if (state.stage === 4 && state.response) {
          this.results = state.response;
          
          // Replace simulated logs with real backend logs if available
          if (state.response.logs && state.response.logs.length > 0) {
            this.logs = state.response.logs.map((l: any) => {
              const now = new Date();
              const timeStr = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}.${now.getMilliseconds().toString().padStart(3, '0')}`;
              return {
                time: timeStr,
                tag: l.tag,
                message: l.message,
                type: l.type || 'info'
              };
            });
          }

          // Show token usage and estimated cost
          if (state.response.tokenUsage) {
            const usage = state.response.tokenUsage;
            if (usage.inputTokens > 0 || usage.outputTokens > 0) {
              const cost = this.getEstimatedCost(usage);
              this.addLog(
                'SYSTEM',
                `Claude Token Usage: ${usage.inputTokens} input tokens, ${usage.outputTokens} output tokens. Estimated Cost: $${cost.toFixed(4)}`,
                'success'
              );
            }
          }
          
          const decision = state.response.routerDecision;
          this.addLog('SYSTEM', `Retrieved ${state.response.results.length} candidate frame(s). Router decision: ${decision.toUpperCase()}. Pipeline execution time: ${state.response.executionTimeMs}ms`, 'success');
        }
      },
      error: (err) => {
        console.error(err);
        this.isInvestigating = false;
        this.investigationSub = null;
        this.addLog('ERROR', `Pipeline execution failed: ${err.message || err}`, 'error');
      },
      complete: () => {
        this.isInvestigating = false;
        this.investigationSub = null;
      }
    });
  }

  getEstimatedCost(tokenUsage: { inputTokens: number; outputTokens: number }): number {
    if (!tokenUsage) return 0;
    return (tokenUsage.inputTokens * 0.000001) + (tokenUsage.outputTokens * 0.000005);
  }
}

