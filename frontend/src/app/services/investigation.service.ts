import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, concat, of, timer } from 'rxjs';
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

  constructor(private http: HttpClient) {}

  /**
   * Orchestrates a hybrid pipeline visualization + real API request.
   * Runs local progress simulations for Stage 1 & 2 to show the architecture working,
   * fires the real API request to FastAPI, and transitions to Stage 3 if escalated.
   */
  investigate(query: string, videoFile: File): Observable<PipelineState> {
    const formData = new FormData();
    formData.append('video', videoFile);
    formData.append('query', query);

    // FastAPI endpoint
    const backendUrl = 'http://localhost:8000/api/investigate';

    // 1. Stage 1 (Edge): 0% to 100% progress simulation
    const stage1Sequence$ = new Observable<PipelineState>(subscriber => {
      let progress = 0;
      const interval = setInterval(() => {
        progress += 10;
        subscriber.next({
          stage: 1,
          progress: progress,
          statusMessage: `Edge Filter: Local frame extraction & CLIP similarity search... (${progress}%)`
        });
        if (progress >= 100) {
          clearInterval(interval);
          subscriber.complete();
        }
      }, 150);
    });

    // 2. Stage 2 (Router): Evaluate threshold
    const stage2Sequence$ = of<PipelineState>({
      stage: 2,
      progress: 100,
      statusMessage: 'Confidence-Gated Router: Evaluating score against thresholds (τ_low = 0.25, τ_high = 0.35)...'
    }).pipe(delay(500));

    // 3. Make real HTTP Request to FastAPI
    // While request is in-flight, show router decision state
    const apiCallAndTransition$ = this.http.post<InvestigationResponse>(backendUrl, formData).pipe(
      switchMap((response: InvestigationResponse) => {
        const decisionMsg = response.routerDecision === 'escalated'
          ? `Router Result: CLIP confidence = ${response.confidenceScore}. Ambiguous bounds. ESCALATING TO CLOUD...`
          : `Router Result: CLIP confidence = ${response.confidenceScore}. High match. Skipping cloud reasoning (GDPR Compliant).`;

        const decisionState$ = of<PipelineState>({
          stage: 2,
          progress: 100,
          statusMessage: decisionMsg
        });

        if (response.routerDecision === 'escalated') {
          // If escalated, show Stage 3 Cloud Reasoning loading state
          const stage3State$ = of<PipelineState>({
            stage: 3,
            progress: 100,
            statusMessage: 'Cloud Reasoner: Escalating candidates to Claude Haiku 4.5...'
          }).pipe(delay(300));

          const stage3Processing$ = of<PipelineState>({
            stage: 3,
            progress: 100,
            statusMessage: 'Cloud Reasoner: Claude validating semantic truth and generating forensic summaries...'
          }).pipe(delay(1200));

          const completedState$ = of<PipelineState>({
            stage: 4,
            progress: 100,
            statusMessage: 'Investigation completed successfully.',
            response: response
          }).pipe(delay(500));

          return concat(decisionState$.pipe(delay(1000)), stage3State$, stage3Processing$, completedState$);
        } else {
          // If skipped, transition directly to complete
          const completedState$ = of<PipelineState>({
            stage: 4,
            progress: 100,
            statusMessage: 'Investigation completed successfully.',
            response: response
          }).pipe(delay(1000));

          return concat(decisionState$, completedState$);
        }
      }),
      catchError((error) => {
        console.error('API call failed, falling back to mock behavior:', error);
        
        // Dynamic mock response for fallback/safety
        const mockResponse: InvestigationResponse = {
          query: query,
          routerDecision: 'escalated',
          confidenceScore: 0.68,
          executionTimeMs: 2450,
          results: [
            {
              timestamp: '00:00:12',
              frameIndex: 12,
              confidence: 0.92,
              imageUrl: 'assets/sample_frame.jpg',
              summary: `(Fallback Demonstration Mode) Claude verified: Subject matching the query '${query}' identified in the scene. Dynamic evaluation metrics remain optimal.`
            }
          ]
        };

        const errorState$ = of<PipelineState>({
          stage: 2,
          progress: 100,
          statusMessage: 'FastAPI Backend unreachable. Falling back to local offline simulation.'
        }).pipe(delay(200));

        const stage3Mock$ = of<PipelineState>({
          stage: 3,
          progress: 100,
          statusMessage: 'Cloud Reasoner (Simulation): Simulating Claude verification...'
        }).pipe(delay(1200));

        const completedState$ = of<PipelineState>({
          stage: 4,
          progress: 100,
          statusMessage: 'Investigation completed (Simulation Mode).',
          response: mockResponse
        }).pipe(delay(500));

        return concat(errorState$, stage3Mock$, completedState$);
      })
    );

    return concat(stage1Sequence$, stage2Sequence$, apiCallAndTransition$);
  }
}
