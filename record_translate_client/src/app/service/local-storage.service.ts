import { Injectable } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class LocalStorageService {

  constructor() { }

  // データをlocalStorageに保存
  setItem(key: string, value: string): void {
    localStorage.setItem(key, value);
  }

  // localStorageからデータを取得
  getItem(key: string): string | null {
    return localStorage.getItem(key);
  }

  // localStorageからデータを削除
  removeItem(key: string): void {
    localStorage.removeItem(key);
  }

  // localStorageを全てクリア
  clear(): void {
    localStorage.clear();
  }
}