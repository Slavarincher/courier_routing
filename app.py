from __future__ import annotations

import csv
import io
import threading
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from es.expert_system import ExpertSystem, PlanError, fmt_time, parse_time

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
app.secret_key = "courier-es-prototype"

ES = ExpertSystem(BASE)
STATE = {"shift": ES.new_shift()}
LOCK = threading.Lock()


@app.template_filter("hhmm")
def _hhmm(m):
    return fmt_time(m)


@app.template_filter("dec")
def _dec(x, digits=1):
    if x is None:
        return ""
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return ("{:." + str(digits) + "f}").format(x).replace(".", ",")


@app.context_processor
def _globals():
    shift = STATE["shift"]
    return {"shift": shift, "routes": list(shift.routes.values()), "has_plan": shift.last_plan is not None}


def shift():
    return STATE["shift"]


def render_index(errors=None, form=None, status=200):
    return render_template("index.html", errors=errors or {}, form=form or {},
                           addresses=ES.known_addresses(), states=ES.traffic_states(),
                           priorities=ES.kb.slots["priority"].allowed_values,
                           kb_summary=ES.kb.summary(), rules_count=len({r.no for r in ES.engine.rules}),
                           productions=len(ES.engine.rules),
                           load_warnings=ES.load_warnings), status


@app.get("/")
def index():
    return render_index()


@app.post("/orders/add")
def add_order():
    with LOCK:
        order, errors = ES.validate_order(request.form.to_dict(), shift())
        if errors:
            return render_index(errors, request.form.to_dict(), 400)
        shift().orders.append(order)
    flash("Заказ {} добавлен.".format(order.id), "ok")
    return redirect(url_for("index"))


@app.post("/orders/<oid>/delete")
def delete_order(oid):
    with LOCK:
        o = shift().order(oid)
        if o and o.active:
            shift().orders.remove(o)
            flash("Заказ {} удален.".format(oid), "ok")
    return redirect(url_for("index"))


@app.post("/orders/upload")
def upload_orders():
    if request.form.get("sample"):
        text = (BASE / "data" / "sample_orders.csv").read_text(encoding="utf-8")
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            flash("Выберите файл CSV.", "error")
            return redirect(url_for("index"))
        raw = f.read()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1251")
    dialect = ";" if text.count(";") >= text.count(",") else ","
    added, problems = [], []
    with LOCK:
        for n, row in enumerate(csv.DictReader(io.StringIO(text), delimiter=dialect), start=2):
            form = {"order_id": row.get("orderId", ""), "address": row.get("address", ""),
                    "priority": row.get("priority", ""), "weight": row.get("weightKg", ""),
                    "window_start": row.get("windowStart", ""), "window_end": row.get("windowEnd", ""),
                    "source": "CSV"}
            order, errors = ES.validate_order(form, shift())
            if errors:
                problems.append("строка {}: {}".format(n, "; ".join(errors.values())))
            else:
                shift().orders.append(order)
                added.append(order.id)
    if added:
        flash("Из CSV добавлено заказов: {} ({}).".format(len(added), ", ".join(added)), "ok")
    for p in problems:
        flash("CSV, " + p, "error")
    return redirect(url_for("index"))


@app.post("/reset")
def reset():
    with LOCK:
        STATE["shift"] = ES.new_shift()
    flash("Данные смены загружены из Базы Знаний заново.", "ok")
    return redirect(url_for("index"))


def apply_plan_form(form) -> list:
    errors = []
    sh = shift()
    start = parse_time(form.get("start_time", ""))
    if start is None:
        errors.append("Время начала смены в формате ЧЧ:ММ")
    else:
        sh.start_min = start
    for c in sh.couriers:
        if sh.route_of(c.id):
            continue
        c.active = form.get("active_" + c.id) == "on"
        rating = (form.get("rating_" + c.id) or "").replace(",", ".")
        e = ES.kb.check_value("courierRating", rating)
        if e:
            errors.append("Рейтинг курьера {}: {}".format(c.short_name, e))
        else:
            c.rating = float(rating)
    for g in sh.segments:
        state = form.get("state_" + g.id, g.state)
        if state in dict(ES.traffic_states()):
            g.state = state
        delay = form.get("delay_" + g.id, str(g.delay))
        e = ES.kb.check_value("delayMin", delay)
        if e:
            errors.append("Задержка на участке «{}»: {}".format(g.name, e))
        else:
            g.delay = int(float(delay))
    return errors


@app.post("/plan")
def plan():
    with LOCK:
        errors = apply_plan_form(request.form)
        if errors:
            for e in errors:
                flash(e, "error")
            return redirect(url_for("index"))
        try:
            ES.plan(shift())
        except PlanError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))
    return redirect(url_for("variants"))


@app.get("/plan")
def variants():
    if shift().last_plan is None:
        return redirect(url_for("index"))
    return render_template("plan.html", result=shift().last_plan)


@app.post("/recalculate")
def recalculate():
    with LOCK:
        try:
            ES.plan(shift())
            flash("Варианты пересчитаны с учетом текущих данных.", "ok")
        except PlanError as e:
            flash(str(e), "error")
            return redirect(url_for("index"))
    return redirect(url_for("variants"))


@app.post("/approve")
def approve():
    with LOCK:
        try:
            route = ES.approve(shift(), request.form["courier_id"], request.form.get("variant", "A"))
        except PlanError as e:
            flash(str(e), "error")
            return redirect(url_for("variants"))
    return redirect(url_for("route_view", rid=route.id))


@app.get("/route/<rid>")
def route_view(rid):
    route = shift().routes.get(rid)
    if route is None:
        flash("Маршрут {} не найден.".format(rid), "error")
        return redirect(url_for("index"))
    return render_template("route.html", route=route, courier=shift().courier(route.courier_id))


@app.post("/route/<rid>/event")
def route_event(rid):
    with LOCK:
        try:
            ES.handle_event(shift(), rid, request.form.get("kind", ""), request.form.to_dict())
        except (PlanError, KeyError) as e:
            flash(str(e), "error")
    return redirect(url_for("route_view", rid=rid))


if __name__ == "__main__":
    print("ЭС «Маршрут курьера»: База Знаний -", ES.kb.summary(), "| правил:", len(ES.engine.rules))
    print("Откройте в браузере: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
