{{/*
Expand the name of the chart.
*/}}
{{- define "automata-app.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "automata-app.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "automata-app.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "automata-app.labels" -}}
helm.sh/chart: {{ include "automata-app.chart" . }}
{{ include "automata-app.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "automata-app.selectorLabels" -}}
app.kubernetes.io/name: {{ include "automata-app.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "automata-app.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "automata-app.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the namespace to use
*/}}
{{- define "automata-app.namespace" -}}
{{- if .Values.namespaceOverride }}
{{- .Values.namespaceOverride }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Backend labels
*/}}
{{- define "automata-app.backend.labels" -}}
{{ include "automata-app.labels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Backend selector labels
*/}}
{{- define "automata-app.backend.selectorLabels" -}}
{{ include "automata-app.selectorLabels" . }}
app.kubernetes.io/component: backend
{{- end }}

{{/*
Frontend labels
*/}}
{{- define "automata-app.frontend.labels" -}}
{{ include "automata-app.labels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Frontend selector labels
*/}}
{{- define "automata-app.frontend.selectorLabels" -}}
{{ include "automata-app.selectorLabels" . }}
app.kubernetes.io/component: frontend
{{- end }}

{{/*
Create a default fully qualified postgresql name.
*/}}
{{- define "automata-app.postgresql.fullname" -}}
{{- printf "%s-%s" .Release.Name "postgresql" | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Create a default fully qualified valkey name.
*/}}
{{- define "automata-app.valkey.fullname" -}}
{{- printf "%s-%s" .Release.Name "valkey" | trunc 63 | trimSuffix "-" -}}
{{- end }}

{{/*
Get the postgres password secret name
*/}}
{{- define "automata-app.postgresql.secretName" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "%s" (include "automata-app.postgresql.fullname" .) -}}
{{- else }}
{{- printf "%s" (include "automata-app.fullname" .) -}}
{{- end }}
{{- end }}

{{/*
Get the valkey password secret name
*/}}
{{- define "automata-app.valkey.secretName" -}}
{{- if .Values.valkey.enabled }}
{{- printf "%s" (include "automata-app.valkey.fullname" .) -}}
{{- else }}
{{- printf "%s" (include "automata-app.fullname" .) -}}
{{- end }}
{{- end }}

{{/*
Database URL for the application
*/}}
{{- define "automata-app.databaseUrl" -}}
{{- if .Values.postgresql.enabled -}}
{{- $host := include "automata-app.postgresql.fullname" . -}}
{{- $port := .Values.postgresql.primary.service.ports.postgresql -}}
{{- $database := .Values.postgresql.auth.database -}}
{{- $username := .Values.postgresql.auth.username -}}
postgresql://{{ $username }}:$(DATABASE_PASSWORD)@{{ $host }}:{{ $port }}/{{ $database }}
{{- else -}}
{{- .Values.externalDatabase.url -}}
{{- end -}}
{{- end }}

{{/*
Valkey URL for the application
*/}}
{{- define "automata-app.valkeyUrl" -}}
{{- if .Values.valkey.enabled -}}
{{- $host := printf "%s-master" (include "automata-app.valkey.fullname" .) -}}
{{- $port := .Values.valkey.master.service.ports.valkey -}}
valkey://{{ $host }}:{{ $port }}
{{- else -}}
{{- .Values.externalValkey.url -}}
{{- end -}}
{{- end }}

{{/*
Common environment variables
*/}}
{{- define "automata-app.commonEnv" -}}
- name: DATABASE_URL
  value: {{ include "automata-app.databaseUrl" . | quote }}
- name: DATABASE_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ include "automata-app.postgresql.secretName" . }}
      key: {{ if .Values.postgresql.enabled }}postgres-password{{ else }}password{{ end }}
- name: VALKEY_URL
  value: {{ include "automata-app.valkeyUrl" . | quote }}
- name: SECRET_KEY
  valueFrom:
    secretKeyRef:
      name: {{ include "automata-app.fullname" . }}-secrets
      key: secretKey
- name: JWT_SECRET
  valueFrom:
    secretKeyRef:
      name: {{ include "automata-app.fullname" . }}-secrets
      key: jwtSecret
{{- end }}

{{/*
Image repository helper
*/}}
{{- define "automata-app.image" -}}
{{- $registry := .Values.global.imageRegistry | default .Values.image.registry -}}
{{- if $registry -}}
{{ $registry }}/{{ .repository }}:{{ .tag | default .Chart.AppVersion }}
{{- else -}}
{{ .repository }}:{{ .tag | default .Chart.AppVersion }}
{{- end -}}
{{- end }}

{{/*
Return the proper Storage Class
*/}}
{{- define "automata-app.storageClass" -}}
{{- if .Values.global.storageClass -}}
{{- if (eq "-" .Values.global.storageClass) -}}
storageClassName: ""
{{- else }}
storageClassName: {{ .Values.global.storageClass | quote }}
{{- end -}}
{{- else if .Values.persistence.storageClass -}}
{{- if (eq "-" .Values.persistence.storageClass) -}}
storageClassName: ""
{{- else }}
storageClassName: {{ .Values.persistence.storageClass | quote }}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Validate required values
*/}}
{{- define "automata-app.validateValues" -}}
{{- if and (not .Values.postgresql.enabled) (not .Values.externalDatabase.url) -}}
automata-app: database
    Either enable PostgreSQL or configure an external database URL
{{- end -}}
{{- if and .Values.ingress.enabled (not .Values.ingress.hosts) -}}
automata-app: ingress
    Ingress is enabled but no hosts are configured
{{- end -}}
{{- end -}}