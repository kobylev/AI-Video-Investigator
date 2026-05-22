import { Injectable, NgZone } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, concat, of, timer, timeout } from 'rxjs';
import { map, delay, switchMap, catchError } from 'rxjs/operators';

export interface InvestigationResult {
  timestamp: string;
  frameIndex: number;
  confidence: number;
  imageUrl: string;
  summary: string;
  routerDecision?: 'accepted' | 'escalated';
}

export interface InvestigationResponse {
  query: string;
  routerDecision: 'skip' | 'escalated';
  confidenceScore: number;
  executionTimeMs: number;
  results: InvestigationResult[];
  tokenUsage?: {
    inputTokens: number;
    outputTokens: number;
  };
  logs?: Array<{
    tag: string;
    message: string;
    type: string;
  }>;
}

export interface PipelineState {
  stage: 1 | 2 | 3 | 4; // 1: Edge, 2: Router, 3: Cloud, 4: Completed
  progress: number;     // 0 - 100%
  statusMessage: string;
  response?: InvestigationResponse;
}

@Injectable({
  providedIn: 'root'
})
export class InvestigationService {

  constructor(private http: HttpClient, private ngZone: NgZone) {}

  /**
   * Orchestrates a hybrid pipeline visualization + real API request.
   * Runs local progress simulations for Stage 1 & 2 to show the architecture working,
   * fires the real API request to FastAPI, and transitions to Stage 3 if escalated.
   */
  investigate(query: string, videoFile: File, tauLow?: number, tauHigh?: number, freeOnly?: boolean, fps?: number): Observable<PipelineState> {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('query', query);
    if (tauLow !== undefined) {
      formData.append('tau_low', tauLow.toString());
    }
    if (tauHigh !== undefined) {
      formData.append('tau_high', tauHigh.toString());
    }
    if (freeOnly !== undefined) {
      formData.append('free_only', freeOnly ? 'true' : 'false');
    }
    if (fps !== undefined) {
      formData.append('fps', fps.toString());
    }

    const backendUrl = 'http://localhost:8000/api/investigate';

    return new Observable<PipelineState>(subscriber => {
      const abortController = new AbortController();
      const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

      // Wrap subscriber methods in ngZone.run to ensure change detection triggers for async fetch stream callbacks
      const originalNext = subscriber.next.bind(subscriber);
      const originalError = subscriber.error.bind(subscriber);
      const originalComplete = subscriber.complete.bind(subscriber);

      (subscriber as any).next = (state: PipelineState) => this.ngZone.run(() => originalNext(state));
      (subscriber as any).error = (err: any) => this.ngZone.run(() => originalError(err));
      (subscriber as any).complete = () => this.ngZone.run(() => originalComplete());

      (async () => {
        try {
          const response = await fetch(backendUrl, {
            method: 'POST',
            body: formData,
            signal: abortController.signal
          });

          if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
          }

          if (!response.body) {
            throw new Error('Response body is null');
          }

          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');
          let buffer = '';

          while (true) {
            const { value, done } = await reader.read();
            if (done) {
              break;
            }

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
              const trimmed = line.trim();
              if (trimmed) {
                try {
                  const state = JSON.parse(trimmed) as PipelineState;
                  subscriber.next(state);
                } catch (e) {
                  console.error('Failed to parse stream line:', trimmed, e);
                }
              }
            }
          }

          if (buffer.trim()) {
            try {
              const state = JSON.parse(buffer.trim()) as PipelineState;
              subscriber.next(state);
            } catch (e) {
              console.error('Failed to parse final stream line:', buffer, e);
            }
          }

          subscriber.complete();
        } catch (err: any) {
          if (err.name === 'AbortError') {
            console.log('Stream fetch aborted.');
            return;
          }

          console.warn('API call failed, falling back to mock behavior:', err);

          try {
            // Simulated Stage 1: Edge Retrieval
            for (let p = 10; p <= 100; p += 10) {
              if (abortController.signal.aborted) return;
              subscriber.next({
                stage: 1,
                progress: p,
                statusMessage: `Edge Filter: Local frame extraction & CLIP similarity search... (${p}%)`
              });
              await delay(150);
            }

            // Simulated Stage 2: Router Gate
            if (abortController.signal.aborted) return;
            subscriber.next({
              stage: 2,
              progress: 100,
              statusMessage: 'FastAPI Backend unreachable. Falling back to local offline simulation.'
            });
            await delay(1000);

            const mockResponse: InvestigationResponse = {
              query: query,
              routerDecision: freeOnly ? 'skip' : 'escalated',
              confidenceScore: freeOnly ? 0.45 : 0.68,
              executionTimeMs: freeOnly ? 920 : 2450,
              results: freeOnly ? [
                {
                  timestamp: '00:00:12',
                  frameIndex: 12,
                  confidence: 0.45,
                  imageUrl: 'assets/sample_frame.jpg',
                  summary: `(Fallback CLIP-Only Mode) Direct CLIP matching: Subject matching the query '${query}' identified in the scene (Score: 0.45).`,
                  routerDecision: 'accepted'
                }
              ] : [
                {
                  timestamp: '00:00:12',
                  frameIndex: 12,
                  confidence: 0.92,
                  imageUrl: 'assets/sample_frame.jpg',
                  summary: `(Fallback Demonstration Mode) Claude verified: Subject matching the query '${query}' identified in the scene. Dynamic evaluation metrics remain optimal.`
                }
              ],
              tokenUsage: freeOnly ? { inputTokens: 0, outputTokens: 0 } : { inputTokens: 258, outputTokens: 55 },
              logs: freeOnly ? [
                { tag: 'INGEST', message: `Received query: '${query}' for video`, type: 'info' },
                { tag: 'INGEST', message: 'Video file saved to local uploads directory.', type: 'success' },
                { tag: 'EDGE', message: 'Loading pre-computed CLIP FAISS index from disk...', type: 'info' },
                { tag: 'EDGE', message: 'Successfully loaded FAISS index.', type: 'success' },
                { tag: 'EDGE', message: `Computing text embeddings for search query: '${query}'`, type: 'info' },
                { tag: 'EDGE', message: 'Searching FAISS index for top 20 candidate matches...', type: 'info' },
                { tag: 'ROUTER', message: `Evaluating candidates using thresholds: tau_low = ${tauLow ?? 0.22}, tau_high = ${tauHigh ?? 0.29}...`, type: 'info' },
                { tag: 'ROUTER', message: 'Router Decision: Free Only Mode active. 1 accepted directly (CLIP > tau_high), 0 escalated to Cloud Reasoner (tau_low < CLIP < tau_high)', type: 'success' },
                { tag: 'SYSTEM', message: 'Pipeline completed (CLIP Only). 1 visual match(es) retrieved in 920ms.', type: 'success' }
              ] : [
                { tag: 'INGEST', message: `Received query: '${query}' for video`, type: 'info' },
                { tag: 'INGEST', message: 'Video file saved to local uploads directory.', type: 'success' },
                { tag: 'EDGE', message: 'Loading pre-computed CLIP FAISS index from disk...', type: 'info' },
                { tag: 'EDGE', message: 'Successfully loaded FAISS index.', type: 'success' },
                { tag: 'EDGE', message: `Computing text embeddings for search query: '${query}'`, type: 'info' },
                { tag: 'EDGE', message: 'Searching FAISS index for top 20 candidate matches...', type: 'info' },
                { tag: 'ROUTER', message: `Evaluating candidates using thresholds: tau_low = ${tauLow ?? 0.22}, tau_high = ${tauHigh ?? 0.29}...`, type: 'info' },
                { tag: 'ROUTER', message: 'Router Decision: 0 accepted directly (CLIP > tau_high), 1 escalated to Cloud Reasoner (tau_low < CLIP < tau_high)', type: 'success' },
                { tag: 'CLOUD', message: 'Simulating Claude validation for frame at 12.0s (Frame #12)...', type: 'info' },
                { tag: 'CLOUD', message: 'Claude verdict: event_detected=True, confidence=0.92', type: 'success' },
                { tag: 'SYSTEM', message: 'Pipeline completed. 1 visual match(es) retrieved in 2450ms.', type: 'success' }
              ]
            };

            if (freeOnly) {
              if (abortController.signal.aborted) return;
              subscriber.next({
                stage: 4,
                progress: 100,
                statusMessage: 'Investigation completed (Simulation Mode - CLIP Only).',
                response: mockResponse
              });
            } else {
              if (abortController.signal.aborted) return;
              subscriber.next({
                stage: 3,
                progress: 100,
                statusMessage: 'Cloud Reasoner (Simulation): Simulating Claude verification...'
              });
              await delay(1200);

              if (abortController.signal.aborted) return;
              subscriber.next({
                stage: 4,
                progress: 100,
                statusMessage: 'Investigation completed (Simulation Mode).',
                response: mockResponse
              });
            }
            subscriber.complete();
          } catch (simErr) {
            subscriber.error(simErr);
          }
        }
      })();

      return () => {
        abortController.abort();
      };
    });
  }
}
