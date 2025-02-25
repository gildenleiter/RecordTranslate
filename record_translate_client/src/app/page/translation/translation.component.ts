import { Component } from '@angular/core';
import { LocalStorageService } from '../../service/local-storage.service';
@Component({
  selector: 'app-translation',
  standalone: true,
  imports: [],
  templateUrl: './translation.component.html',
  styleUrl: './translation.component.css'
})
export class TranslationComponent {
  sessionInfo: string | null = '';
  private intervalId: any;

  constructor(private localStorageService: LocalStorageService) {}

  ngOnInit() {
    this.sessionInfo = this.localStorageService.getItem('Translation');
    // 2秒ごとに翻訳結果を取得し、画面を更新
    this.intervalId = setInterval(() => {
      this.sessionInfo = this.localStorageService.getItem('Translation');
    }, 2000);  // 2000ms = 2秒
  }

  ngOnDestroy(): void {
    // コンポーネント破棄時にsetIntervalをクリア
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
  }
}
