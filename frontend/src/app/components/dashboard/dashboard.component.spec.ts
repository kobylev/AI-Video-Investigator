import { TestBed, ComponentFixture } from '@angular/core/testing';
import { describe, it, expect, beforeEach, beforeAll, vi } from 'vitest';
import { ReactiveFormsModule, FormBuilder, FormsModule } from '@angular/forms';
import { HttpClientTestingModule } from '@angular/common/http/testing';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { NO_ERRORS_SCHEMA } from '@angular/core';

// Angular Material Imports
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatCardModule } from '@angular/material/card';
import { MatMenuModule } from '@angular/material/menu';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatListModule } from '@angular/material/list';
import { MatDividerModule } from '@angular/material/divider';
import { MatTooltipModule } from '@angular/material/tooltip';
import { MatSliderModule } from '@angular/material/slider';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatStepperModule } from '@angular/material/stepper';
import { MatCheckboxModule } from '@angular/material/checkbox';

import { DashboardComponent } from './dashboard.component';
import { InvestigationService } from '../../services/investigation.service';
import { of } from 'rxjs';

describe('DashboardComponent', () => {
  let component: DashboardComponent;
  let fixture: ComponentFixture<DashboardComponent>;
  let mockInvestigationService: any;

  beforeAll(() => {
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: vi.fn().mockImplementation(query => ({
        matches: false,
        media: query,
        onchange: null,
        addListener: vi.fn(),
        removeListener: vi.fn(),
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
      })),
    });
  });

  beforeEach(async () => {
    mockInvestigationService = {
      investigate: vi.fn().mockReturnValue(of({ stage: 1, progress: 0, statusMessage: 'Starting...' }))
    };

    await TestBed.configureTestingModule({
      declarations: [DashboardComponent],
      imports: [
        ReactiveFormsModule,
        FormsModule,
        HttpClientTestingModule,
        BrowserAnimationsModule,
        MatToolbarModule,
        MatIconModule,
        MatButtonModule,
        MatFormFieldModule,
        MatInputModule,
        MatProgressBarModule,
        MatCardModule,
        MatMenuModule,
        MatProgressSpinnerModule,
        MatListModule,
        MatDividerModule,
        MatTooltipModule,
        MatSliderModule,
        MatExpansionModule,
        MatStepperModule,
        MatCheckboxModule
      ],
      providers: [
        FormBuilder,
        { provide: InvestigationService, useValue: mockInvestigationService }
      ]
    }).compileComponents();

    fixture = TestBed.createComponent(DashboardComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create the dashboard component', () => {
    expect(component).toBeTruthy();
  });

  it('should initialize with default thresholds', () => {
    expect(component.tauLow).toBe(0.22);
    expect(component.tauHigh).toBe(0.29);
  });

  it('should update tauLow and cap tauHigh if tauLow > tauHigh', () => {
    component.onTauLowChange(0.35);
    expect(component.tauLow).toBe(0.35);
    expect(component.tauHigh).toBe(0.35);
  });

  it('should update tauHigh and cap tauLow if tauHigh < tauLow', () => {
    component.onTauHighChange(0.15);
    expect(component.tauHigh).toBe(0.15);
    expect(component.tauLow).toBe(0.15);
  });

  it('should reset expert settings to defaults', () => {
    component.tauLow = 0.5;
    component.tauHigh = 0.6;
    component.resetExpertSettings();
    expect(component.tauLow).toBe(0.22);
    expect(component.tauHigh).toBe(0.29);
  });

  it('should add diagnostic logs', () => {
    const initialLogsCount = component.logs.length;
    component.addLog('TEST', 'Test message', 'info');
    expect(component.logs.length).toBe(initialLogsCount + 1);
    expect(component.logs[component.logs.length - 1].tag).toBe('TEST');
    expect(component.logs[component.logs.length - 1].message).toBe('Test message');
  });
});
