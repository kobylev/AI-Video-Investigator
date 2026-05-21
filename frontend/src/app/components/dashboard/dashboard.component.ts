import { Component, HostBinding, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
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
  dragOver = false;
  
  isDarkTheme = true; // Default to modern Dark Theme
  currentState: PipelineState | null = null;
  isInvestigating = false;
  results: InvestigationResponse | null = null;

  @HostBinding('class.dark-theme') get themeClass() {
    return this.isDarkTheme;
  }

  constructor(
    private fb: FormBuilder,
    private investigationService: InvestigationService
  ) {}

  ngOnInit(): void {
    this.investigationForm = this.fb.group({
      query: ['', [Validators.required, Validators.minLength(3)]]
    });
    
    // Apply dark theme by default to the body element
    document.body.classList.add('dark-theme');
    document.body.classList.remove('light-theme');
  }

  toggleTheme(): void {
    this.isDarkTheme = !this.isDarkTheme;
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
      this.selectedFile = input.files[0];
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
      }
    }
  }

  clearFile(): void {
    this.selectedFile = null;
  }

  startInvestigation(): void {
    if (this.investigationForm.invalid || !this.selectedFile) return;

    this.isInvestigating = true;
    this.results = null;
    this.currentState = null;

    const query = this.investigationForm.value.query;
    
    this.investigationService.investigate(query, this.selectedFile).subscribe({
      next: (state: PipelineState) => {
        this.currentState = state;
        if (state.stage === 4 && state.response) {
          this.results = state.response;
        }
      },
      error: (err) => {
        console.error(err);
        this.isInvestigating = false;
      },
      complete: () => {
        this.isInvestigating = false;
      }
    });
  }
}
