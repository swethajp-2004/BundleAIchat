# utils.py — corrected for headless server use
import io
import os
import base64
import matplotlib

# Force non-GUI backend before importing pyplot (important on servers)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union, Iterable
from html import escape

# Optional OpenAI wrapper for embeddings used by server; requires OPENAI_API_KEY in env.
try:
    from openai import OpenAI
    _OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    _openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
except Exception:
    _openai_client = None

def embed_text(texts: Union[str, List[str]], model: str = "text-embedding-3-small"):
    if _openai_client is None:
        raise RuntimeError("OpenAI client not configured. Set OPENAI_API_KEY in environment or remove embed_text usage.")
    single = False
    if isinstance(texts, str):
        texts = [texts]
        single = True
    try:
        resp = _openai_client.embeddings.create(model=model, input=texts)
        embs = [r.embedding for r in resp.data]
        return embs[0] if single else embs
    except Exception as e:
        raise RuntimeError(f"Embedding request failed: {e}")

def _safe_float_list(values):
    out = []
    for v in values:
        try:
            out.append(float(v))
        except Exception:
            try:
                out.append(float(str(v).replace(",", "")))
            except Exception:
                continue
    return out

def plot_to_base64(x: Iterable, y: Union[Iterable[float], Iterable[Iterable[float]]], kind='line', title=''):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        if kind == 'line':
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            ax.plot(labels, vals, marker='o', linewidth=2)
            for xi, yi in zip(labels, vals):
                try:
                    ax.annotate(f"{yi:,.0f}", (xi, yi), textcoords="offset points", xytext=(0,6), ha='center', fontsize=8)
                except Exception:
                    pass
            ax.set_ylabel("Value")
            ax.set_xlabel("")
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

        elif kind == 'bar':
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            ax.bar(labels, vals)
            for rect in ax.patches:
                h = rect.get_height()
                try:
                    ax.annotate(f"{h:,.0f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                                xytext=(0, 6), textcoords="offset points", ha='center', va='bottom', fontsize=8)
                except Exception:
                    pass
            plt.xticks(rotation=30, ha='right')
            ax.set_ylabel("Value")
            plt.tight_layout()

        elif kind == 'pie':
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            if not vals:
                vals = [1] * len(labels)
            n = min(len(labels), len(vals))
            labels = labels[:n]
            vals = vals[:n]
            ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=140)
            ax.axis('equal')
            plt.tight_layout()

        elif kind == 'scatter':
            vals_x = _safe_float_list(x)
            vals_y = _safe_float_list(y)
            ax.scatter(vals_x, vals_y, alpha=0.7)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            plt.tight_layout()

        elif kind == 'stacked_bar':
            labels = list(map(str, x))
            if not hasattr(y[0], "__iter__") or isinstance(y[0], (str, bytes)):
                vals = _safe_float_list(y)
                ax.bar(labels, vals)
            else:
                series = []
                for series_vals in y:
                    series.append(_safe_float_list(series_vals))
                n = len(labels)
                for i in range(len(series)):
                    if len(series[i]) < n:
                        series[i] = series[i] + [0.0] * (n - len(series[i]))
                    else:
                        series[i] = series[i][:n]
                bottoms = [0.0] * n
                for s in series:
                    ax.bar(labels, s, bottom=bottoms)
                    bottoms = [a + b for a, b in zip(bottoms, s)]
                plt.xticks(rotation=30, ha='right')
                plt.tight_layout()

        elif kind == 'hist':
            vals = _safe_float_list(y)
            ax.hist(vals, bins=20)
            ax.set_ylabel("Count")
            ax.set_xlabel(title or "Value")
            plt.tight_layout()

        else:
            labels = list(map(str, x))
            vals = _safe_float_list(y)
            ax.plot(labels, vals, marker='o')
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()

        ax.set_title(title or "")
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('ascii')
        plt.close(fig)
        return f"data:image/png;base64,{img_b64}"
    except Exception:
        # ensure figure closed on error
        try:
            plt.close(fig)
        except Exception:
            pass
        raise

def rows_to_html_table(records):
    if not records:
        return "<p><i>No rows</i></p>"

    headers = list(records[0].keys())

    def fmt(val):
        if val is None:
            return ""
        if isinstance(val, (int, float)) and (abs(val) >= 1 or val == 0):
            try:
                if float(val).is_integer():
                    return f"{int(val):,}"
                else:
                    return f"{float(val):,.2f}"
            except Exception:
                pass
        return escape(str(val))

    html = ['<div class="table-responsive"><table class="table" style="width:100%; border-collapse: collapse;">']
    html.append("<thead><tr>")
    for h in headers:
        html.append(f"<th style='text-align:left; padding:8px; border-bottom:1px solid #ddd;'>{escape(h)}</th>")
    html.append("</tr></thead>")
    html.append("<tbody>")
    for r in records:
        html.append("<tr>")
        for h in headers:
            html.append(f"<td style='padding:8px; border-bottom:1px solid #eee;'>{fmt(r.get(h))}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "".join(html)
