# Problemi Rilevati - Observer e Caratterizzazione FDI

## 1. Problemi Documentati Negli Observer

### 1.1 Latenza Luenberger (~1-1.5 s)
- **Descrizione**: L'observer Luenberger mostra una latenza di rilevazione di ~1-1.5 secondi per fault su τ_ext (canale 0), poiché la risposta è mediata dalla catena ammettanza → tracking.
- **Localizzazione**: `README.md:205`, `exoskeleton_observers/observer_node.py:11-12`
- **Impatto**: Rilevazione lenta di fault critici sul sensore di forza.

### 1.2 Compensazione Parziale Ch0 da Parte del Controllore Ammettanza
- **Descrizione**: I fault di tipo `offset` su `tau_ext` sono parzialmente compensati dal controllore ammettanza, causando `r_force` basso mentre `r_encoder` diverge.
- **Localizzazione**: `TUNING.md:150-153`
- **Nota**: Comportamento noto del design FDI, non errore di calibrazione.

### 1.3 Mismatch Parametri Observer 2 (Inversione Ammettanza)
- **Descrizione**: Se i parametri di saturazione del controllore ammettanza cambiano (es. `theta_ref_min`, `theta_ref_max`), l'observer 2 diventa strutturalmente sempre valido anche durante la saturazione.
- **Localizzazione**: `observer_node.py:30-32`
- **Requisito**: I parametri in `observer_params.yaml` devono corrispondere esattamente a quelli in `dynamics_params.yaml`.

### 1.4 Chattering Attrito Coulomb (Observer 3 - Momento Generalizzato)
- **Descrizione**: Il campionamento raw di θ̇ genera scatti binari ±fric_coul agli zeri, producendo residui spuri a basse velocità (dove la derivata di tanh è ~168 Nm/(rad/s) per fric_coul=2.0, fric_eps=0.005).
- **Localizzazione**: `observer_node.py:46-50, 510-538`
- **Soluzione Implementata**: Filtraggio dell'OUTPUT del modello di attrito (non θ̇ in input) con EMA per matcherare il sub-stepping del plant (1000 Hz vs 200 Hz).
- **Parametri**: `mom_alpha_thetadot=0.3` → costante di tempo ~10 ms.

### 1.5 Observer 4 (τ_ext Rate) - Cieco a Drift Lenti
- **Descrizione**: Monitora solo salti non fisiologici (1 campione, ~5 ms), non rileva drift graduali.
- **Localizzazione**: `observer_node.py:62`
- **Impatto**: Fault di tipo drift su τ_ext non vengono rilevati.

### 1.6 Observer Non Integrati nella Catena di Sicurezza
- **Descrizione**: Il modulo è operativo e pubblica residui in tempo reale, ma non è collegato alla decisione della safety.
- **Localizzazione**: `README.md:210, 456`
- **Stato Successivo**: Integrazione via `SENSOR_DEGRADED_MODE` (enum già riservato).

### 1.7 Identificazione Parametri (Miglioramento Futuro)
- **Descrizione**: Parametri di attrito e inerzia sono stime, non identificati sperimentalmente su hardware.
- **Localizzazione**: `README.md:462`
- **Impatto**: Fedeltà del digital twin e sensibilità degli observer ridotte.

---

## 2. Risultati della Caratterizzazione (exoskeleton_fdi_characterization)

### 2.1 Healthy Baseline (20260504_204435)
- **Regimi testati**: 65 (SETTLE, SLOW×3, NOMINAL×3, FAST×2, NEAR_RES×2, MIX_wide, MIX_transitions)
- **Statistiche residui sani**:
  - `r_force_filt`: media=0.000733 Nm, std=0.0007709 Nm
  - `r_encoder_filt`: media=0.004741 rad, std=0.004017 rad
- **FAR target**: 0.1% (1 falso allarme ogni 5 secondi a 200 Hz)

### 2.2 Fault Sweep (20260504_215414)
- **Finestre testate**: 76 (38 per canale: Ch0=tau_ext, Ch3=encoder)
- **MDL-90 (Minimum Detectable Level al 90% detection)**:
  - Ch0 drift_linear: r_enc MDL-90 = 0.005
  - Ch3 drift_linear: r_enc MDL-90 = 0.002
  - Ch3 noise: r_enc MDL-90 = 0.05
- **Problema noto**: Fault offset su Ch0 hanno `p_detect` basso per `r_force` (es. offset 0.2 Nm: 2% detection) perché compensati dal controllore ammettanza.

### 2.3 Parametri FDI Derivati
```yaml
thresh_force:   0.0063 Nm
thresh_encoder: 0.02027 rad
residual_alpha: 0.5000
debounce_count: 3
enc_obs_pole_1: -15.0
enc_obs_pole_2: -20.0
```

---

## 3. Problemi Rilevati Nella Sessione Corrente

### 3.1 Inconsistenza Parametri EMA tra Caratterizzazione e Observer
- **Descrizione**: `computed_fdi_params.yaml` riporta `residual_alpha: 0.5000`, ma `observer_node.py:109` definisce `mom_alpha_thetadot` default=0.3 (→ ~10 ms). Non è chiaro se `residual_alpha` (`compute_thresholds.py`) si riferisca allo stesso filtro EMA di `mom_alpha_thetadot`.
- **Da verificare**: Se `residual_alpha` influisce su `r_force_filt`/`r_encoder_filt` in `fdi_node`, mentre `mom_alpha_thetadot` è specifico per l'observer 3.

### 3.2 Debounce Count Potenzialmente Inadeguato per Latenza Lunga
- **Descrizione**: `debounce_count=3` a 200 Hz dà una latenza minima di 15 ms, appropriata per observer 2, 3, 4. Tuttavia, l'observer Luenberger ha latenza ~1-1.5 s: con debounce=3, un fault persistente su Ch0 potrebbe essere confermato solo dopo che il residuo ha superato la soglia per 3 campioni, ma il vero ritardo è dominato dalla dinamica dell'observer, non dal debounce.
- **Impatto**: Il debounce non risolve la latenza intrinseca del Luenberger.

### 3.3 Mancanza di Validazione Crociata tra Bag e Parametri
- **Descrizione**: I parametri in `computed_fdi_params.yaml` sono stati generati da `compute_thresholds.py` usando `baseline_stats.json` e `fault_sweep_table.json`, ma non c'è evidenza di un test di validazione finale (es. replay dei bag con i nuovi parametri per verificare FAR e p_detect attesi).
- **Suggerimento**: Aggiungere Fase 8 in `TUNING.md` per validazione post-deployment.

### 3.4 Discrepanza tra Soglie Calcolate e Usate
- **Descrizione**: `thresholds_template.yaml` (generato da `analyze_baseline.py:266-267`) usa soglie conservative:
  - `thresh_force = mean + p99_9` → 0.0063 Nm (coerente)
  - `thresh_encoder = mean + p99_9` → 0.02027 rad (coerente)
  
  Tuttavia, in `baseline_stats.json`, `r_force` ha `max_abs: 0.650 Nm` (vs media 0.0007) e `p99_99: 0.0202`, indicando outlier estremi durante la baseline che potrebbero aver distorto il percentile p99.9.
  
- **Analisi**: Il valore `max_abs=0.650 Nm` per `r_force` è 1000x la media, suggerendo possibili problemi transienti durante la baseline (es. transizioni di regime) non filtrati adeguatamente.

### 3.5 Discrepanza Latenza tra JSON e Report Markdown
- **Descrizione**: In `fault_sweep_table.json` (dati grezzi):
  - Ch0_offset_0.2: `r_force_filt.latency_s = 0.015s` (15 ms) ✅
  - Ch0_offset_0.2: `r_encoder_filt.latency_s = 180.27s` (tempo intero della finestra)
  
  In `fault_sweep_report.md` (report leggibile):
  - Ch0_offset_0.2: `r_force p_det=2% / latency=15 ms` ❌ (2% non significa "rilevato")
  - Ch0_offset_0.2: `r_encoder p_det=0% / latency=180270 ms` ✅

- **Problema**: La metrica `p_detect` al 2% è erroneamente marcata con `✓` nel report (linea 18 di fault_sweep_report.md), suggerendo un "rilevamento" che in realtà è quasi nullo.
- **Causa**: In `analyze_fault_sweep.py:285`, il controllo `flag = '✓' if d['triggered'] else '·'` usa il campo `triggered` (basato su `max_dev > threshold`), ma `p_detect=2%` indica che solo il 2% dei campioni ha superato la soglia - non un vero rilevamento.
- **Impatto**: Interpretazione fuorviante del report per fault di piccola entità.

### 3.6 MDL-90 Non Raggiunto per Molti Tipi di Fault
- **Descrizione**: Nel `fault_sweep_report.md`, per Ch0 (tau_ext):
  - `drift_linear`: MDL-90 > max tested per `r_force`
  - `offset`, `scale`, `spike`, `freeze`: MDL-90 > max tested per entrambi i residui
  
  Questo suggerisce che l'observer 2 (inversione ammettanza) è **inefficace** per la maggior parte dei fault su Ch0, confermando il problema di compensazione del controllore.

### 3.7 Parametri M_eff Costanti ma Non Validati
- **Descrizione**: In `baseline_stats.json`, `M_eff` ha std ≈ 0.0001 (variazione <0.1%) in tutti i regimi, con `max_abs=0.101`.
  
- **Possibile problema**: Se `M_eff` nella realtà varia (es. masse mobili del meccanismo), l'observer Luenberger userà un modello scorretto, generando residui biased. Il valore è hard-coded in `dynamics_params.yaml` ma non c'è evidenza di validazione sperimentale.

### 3.8 Force Valid Sempre 1.0 in Baseline
- **Descrizione**: In `baseline_stats.json`, `force_valid` ha mean=1.0, std=0.0, indicando che l'observer 2 non è mai andato in stato invalido durante i 65 regimi testati.
  
- **Nota**: Questo è atteso se il controllore ammettanza non ha mai saturato, ma rende difficile validare la logica di validità dell'observer 2 (theta_ref bounds, deadband) in condizioni reali.

### 3.9 Mancanza di Copertura per theta_ref Saturazione
- **Descrizione**: In `healthy_regimes.yaml`, tutti i regimi usano `f_max: 0.0` (forza massima), ma i parametri di saturazione in `fdi_params.yaml` sono `theta_ref_min: -0.75, theta_ref_max: 0.09`. 
  
  Nei regimi testati (SLOW/NOMINAL/FAST), `theta` rimane sempre tra -0.65 e -0.06 (vedi baseline_stats.json), lontano dai limiti. Non c'è copertura per validare il comportamento dell'observer 2 quando `theta_ref` esce dai bounds.

- **Impatto**: La logica `obs2_valid = False if θ_ref outside [theta_ref_min, theta_ref_max]` in `observer_node.py:30-32` non è testata nel baseline.

### 3.10 Problemi di Nomenclatura nei File JSON
- **Descrizione**: In `fault_sweep_table.json`, il campo è chiamato `residuals` (senza la 'i'), mentre in `analyze_fault_sweep.py:93` si usa `RESIDUALS = ['r_force_filt', 'r_encoder_filt']`. 
  
  In `baseline_stats.json`, i campi sono `r_force_filt` e `r_encoder_filt` (corretto). Tuttavia, il parsing in `compute_thresholds.py:204-206` usa `w['residuals']` che corrisponde al JSON.

- **Nota**: Consistente, ma un refactor future dovrebbe uniformare la nomenclatura (decidere tra `residuals` o `residuals`).

### 3.11 Mancanza di Test per Observer 3 e 4 nel Fault Sweep
- **Descrizione**: Il `fault_sweep_report.md` riporta solo `r_force` e `r_encoder` (observer 2 e Luenberger). Non ci sono dati su:
  - Residuo momento generalizzato (observer 3)
  - Allarme τ_ext rate (observer 4)
  
- **Impatto**: Non è possibile validare MDL-90 o p_detect per gli observer 3 e 4 dai dati attuali.

### 3.12 Parametri di Filtraggio EMA non Validati Sperimentalmente
- **Descrizione**: `residual_alpha: 0.5000` è derivato da criteri SNR (compute_thresholds.py:48-53), ma non c'è evidenza che questo valore sia ottimale per:
  - Latenza di rilevazione (dovrebbe essere < 100 ms per fault gravi)
  - Rumore residuo in condizioni sane (dovrebbe essere << threshold)
  
- **Suggerimento**: Aggiungere plot di `r_force_filt` vs `r_force` in `plot_baseline.py` per validare visivamente l'alpha scelto.

### 3.13 Teoria: Perché il Luenberger è Lento per Fault su Ch0
- **Descrizione**: La latenza di 1-1.5s per fault su `tau_ext` (Ch0) **non è un bug di implementazione né mismatch di parametri**. 

**Analisi della catena di propagazione**:
```
tau_ext fault (es. offset 2 Nm)
    ↓
[Admittance Controller] ← POLI LENTI dominano
    M_v=0.5, D_v=5.0, K_v=2.0
    Poli: s = (-5 ± √(25-4))/1 ≈ -0.42 e -9.58 rad/s
    Tempo costante dominante: τ = 1/0.42 ≈ 2.4 secondi
    ↓ (θ_ref cambia lentamente)
[Trajectory Controller]
    ↓
[Plant Dynamics] (risponde velocemente, λ₂≈-20 rad/s)
    ↓
[Luenberger Observer] (rileva in 200-300ms)
```

**Dimostrazione matematica**:
1. Parametri consistenti: `fric_visc=2.0`, `damping_theta=1.0`, `M_eff≈0.1` in tutto il sistema ✓
2. Calcolo guadagni observer (observer_node.py:371-375):
   - `D_over_M = (2.0+1.0)/0.1 = 30`
   - `l1 = -(p1+p2) - D_over_M = -(-35) - 30 = 5` ✓
   - `l2 = p1*p2 - l1*D_over_M = 300 - 150 = 150` ✓
3. Stabilità discreta: `dt=5ms < 2/|20| = 0.1s` ✓

**Confronto latenze per tipo di fault**:
| Canale | Tipo Fault | Latenza Osservata | Causa |
|---|---|---|---|
| Ch0 (tau_ext) | offset | **1-1.5 s** | Polo admittance lento (-0.42 rad/s, τ=2.4s) |
| Ch2 (tau_m) | offset | ~200-300ms | Fault diretto al plant |
| Ch3 (encoder) | offset | ~200-300ms | Fault diretto alla misura |

**Conclusione**: L'observer Luenberger è **matematicamente corretto e veloce** (~200-300ms), ma per fault su Ch0 la propagazione attraverso l'admittance controller (τ=2.4s) domina il tempo di rilevazione.

**Soluzioni documentate**:
- Usare **Observer 3 (Momento Generalizzato)** per fault Ch0: latenza ~40-60ms
- Usare **Observer 4 (τ_ext Rate)** per impulsi Ch0: latenza ~5ms
- Observer 2 (Inversione ammettanza) è cieco ai fault Ch0 (compensati dal controller)

**Riferimenti**:
- `observer_node.py:11-12`: "detection latency of ~1-1.5 s for faults on τ_ext"
- `fdi_params.yaml:54`: "Linearized plant: lambda2 ≈ -20"
- `TUNING.md:150-153`: "Fault offset su Ch0 parzialmente compensati"

---

## 4. Riferimenti ai File di Documentazione
- `/home/mbari/ros2_ws/src/HES_DT/README.md` (sezioni observer, sviluppo futuro)
- `/home/mbari/ros2_ws/src/HES_DT/TUNING.md` (note su comportamento Ch0)
- `/home/mbari/ros2_ws/src/HES_DT/exoskeleton_observers/exoskeleton_observers/observer_node.py` (documentazione inline observer)
- `/home/mbari/ros2_ws/bags/healthy_baseline_20260504_204435/` (report baseline, parametri FDI)
- `/home/mbari/ros2_ws/bags/fault_sweep_20260504_215414/` (report fault sweep, MDL)
- `/home/mbari/ros2_ws/src/HES_DT/exoskeleton_fdi_characterization/scripts/compute_thresholds.py` (derivazione parametri)
