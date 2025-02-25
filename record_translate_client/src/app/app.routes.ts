import { Routes } from '@angular/router';
import { RecordComponent } from './page/record/record.component'
import { TranslationComponent } from './page/translation/translation.component'

export const routes: Routes = [
    { path: '', component: RecordComponent },
    { path: 'translation', component: TranslationComponent },
];
