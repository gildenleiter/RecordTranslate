import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { HttpClientModule } from '@angular/common/http';
import { ChangeDetectorRef } from '@angular/core';
import { LocalStorageService } from '../../service/local-storage.service';

@Component({
  selector: 'app-record',
  standalone: true,
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './record.component.html',
  styleUrls: ['./record.component.css']
})

export class RecordComponent {
  audioDevices: MediaDeviceInfo[] = [];
  selectedDeviceId: string | undefined;
  transcription: string = '';
  translation: string = '';
  mediaRecorder: MediaRecorder | undefined;
  audioChunks: Blob[] = [];
  isLoading: boolean = true;
  recordingInterval: number = 5000; // 録音間隔を5秒に設定
  recordingTimer: any;
  isRecording: boolean = false; // 録音が現在進行中かどうかを示すフラグ
  isRestarting: boolean = false; // 録音再開のフラグ

  data: any;
  private apiUrl = 'http://localhost:8000'; // FastAPIサーバーのベースURL

  constructor
  (
    private http: HttpClient, private cdr: ChangeDetectorRef,
    private localStorageService: LocalStorageService
  ){}

  ngOnInit() {
    // セッションをクリア
    localStorage.clear();
    
    if (typeof window !== 'undefined' && typeof navigator !== 'undefined') {
      this.getAudioDevices().then(() => {
        this.isLoading = false;
      });
    } else {
      console.error('Navigator is not available.');
      this.isLoading = false;
    }
  }
  
  getAudioDevices(): Promise<void> {
    return new Promise((resolve, reject) => {
      navigator.mediaDevices.enumerateDevices()
        .then(devices => {
          this.audioDevices = devices.filter(device => device.kind === 'audioinput');
          if (this.audioDevices.length > 0) {
            this.selectedDeviceId = this.audioDevices[0].deviceId;
          }
          resolve();
        })
        .catch(error => {
          console.error('デバイスの取得エラー:', error);
          reject(error);
        });
    });
  }

  startRecording() {
    if (this.isRecording) {
      console.warn('Recording is already in progress.');
      return; // すでに録音が開始されている場合は何もしない
    }

    this.isRecording = true;
    this.audioChunks = []; // 録音開始時に音声チャンクをリセット
    this.isRestarting = false; // 再起動フラグをリセット

    const constraints = {
      audio: {
        deviceId: this.selectedDeviceId ? { exact: this.selectedDeviceId } : undefined
      }
    };

    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        this.mediaRecorder = new MediaRecorder(stream);
        this.mediaRecorder.ondataavailable = event => {
          this.audioChunks.push(event.data);
        };

        this.mediaRecorder.start();
        this.recordingTimer = setTimeout(() => {
          this.stopRecording(true); // 録音を停止し、新たな録音を開始
        }, this.recordingInterval);
      })
      .catch(error => {
        console.error('録音エラー:', error);
        this.isRecording = false; // エラー発生時にはフラグをリセット
      });
  }

  stopRecording(restart: boolean = false) {
    if (this.mediaRecorder && this.isRecording) {
      this.isRecording = false; // 録音フラグをリセットして、複数回呼ばれるのを防ぐ
      this.mediaRecorder.stop();

      this.mediaRecorder.onstop = () => {
        if (this.audioChunks.length > 0 && !this.isRestarting) {
          const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
          // this.downloadAudio(audioBlob);
          // this.getData();
          this.sendAudio(audioBlob);
          this.audioChunks = []; // ダウンロード後にチャンクをクリア
        }

        if (restart && !this.isRestarting) {
          this.isRestarting = true; // 再起動フラグを設定
          this.startRecording(); // 録音を再開
        } else {
          this.mediaRecorder = undefined; // mediaRecorderをリセット
        }
      };
    }
  }

  sendAudio(audioBlob: Blob) {
    const formData = new FormData();
    formData.append('file', audioBlob, `audio_${Date.now()}.wav`);
  
    const headers = new HttpHeaders({
      'Accept': 'application/json',
    });
  
    this.http.post<any>(`${this.apiUrl}/upload`, formData, { headers }).subscribe(
      (response) => {
        // サーバーからのレスポンスを処理
        if (response.transcription && response.translation) {
          this.transcription = response.transcription;
          this.translation = response.translation;
          console.log('Transcription:', this.transcription);
          console.log('Translation:', this.translation);
          
          // セッションに翻訳結果を保存
          this.localStorageService.setItem('Translation', this.translation);

          // this.saveTextAsFile(this.translation);

          // 変更検出を手動でトリガー
          this.cdr.detectChanges();
        } else {
          this.transcription = '';
          this.translation = '';
          // セッションに翻訳結果を保存
          this.localStorageService.setItem('Translation', '');

          // this.saveTextAsFile(this.translation);

          // 変更検出を手動でトリガー
          this.cdr.detectChanges();
          console.error('Invalid response from server:', response);
        }
      },
      (error) => {
        this.transcription = '';
        this.translation = '';
          // セッションに翻訳結果を保存
          this.localStorageService.setItem('Translation', '');

          // this.saveTextAsFile(this.translation);

          // 変更検出を手動でトリガー
          this.cdr.detectChanges();
        console.error('Error uploading audio:', error);
      }
    );
  }

  downloadAudio(audioBlob: Blob) {
    const url = URL.createObjectURL(audioBlob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    a.download = `audio_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
  }

  stopRecordingCycle() {
    if (this.recordingTimer) {
      clearTimeout(this.recordingTimer);
      this.recordingTimer = null; // タイマーをリセット
      this.stopRecording(); // 最後の録音を停止
    }
  }

  processAudio(audioBlob: Blob) {
    this.transcription = '仮の文字起こし結果';
    this.translation = '仮の翻訳結果';
  }

  getData(): void {
    this.http.get<any>(`${this.apiUrl}/`).subscribe(
      (response) => {
        this.data = response;
        console.log(this.data);
      },
      (error) => {
        console.error('Error fetching data:', error);
      }
    );
  }

  translate(): void {
    this.http.get<any>(`${this.apiUrl}/translate`).subscribe(
      (response) => {
        this.data = response;
        console.log(this.data);
      },
      (error) => {
        console.error('Error fetching data:', error);
      }
    );
  }

  // 任意の文字列をテキストファイルとして保存するメソッド
  saveTextAsFile(text:string) {
    const blob = new Blob([text], { type: 'text/plain' });
    
    const link = document.createElement('a');
    link.href = window.URL.createObjectURL(blob);
    link.download = 'myfile.txt'; // ファイル名
    link.click();
    
    window.URL.revokeObjectURL(link.href); // メモリ解放
  }

}