const fileInput = document.getElementById('fileInput');
const loading = document.getElementById('loading');
const error = document.getElementById('error');
const fileListContainer = document.getElementById('fileListContainer');
const fileList = document.getElementById('fileList');
const fileContentView = document.getElementById('fileContentView');
const factMatrixView = document.getElementById('factMatrixView');
const metadataDiv = document.getElementById('metadata');
const pagesDiv = document.getElementById('pages');
const factTableBody = document.getElementById('factTableBody');
const conflictsSection = document.getElementById('conflictsSection');
const conflictsContent = document.getElementById('conflictsContent');
const summaryContent = document.getElementById('summaryContent');

// Store current facts data
let currentFactsData = null;
let currentLiabilitySignalsData = null;
let currentLiabilityRecommendationData = null;
let currentTimelineData = null;
let currentClaimRationaleData = null;
let currentEscalationPackageData = null;

// Predefined list of files to upload
const expectedFiles = [
    { name: 'accident_images.png', type: 'image', displayName: 'Accident Images' },
    { name: 'fnol.pdf', type: 'pdf', displayName: 'First Notice of Loss' },
    { name: 'claimant_statement.pdf', type: 'pdf', displayName: 'Claimant Statement' },
    { name: 'other_driver_statement.pdf', type: 'pdf', displayName: 'Other Driver Statement' },
    { name: 'police_report.pdf', type: 'pdf', displayName: 'Police Report' },
    { name: 'repair_estimate.pdf', type: 'pdf', displayName: 'Repair Estimate' },
    { name: 'state_negligence_rules.pdf', type: 'pdf', displayName: 'State Negligence Rules' },
    { name: 'policy_document.pdf', type: 'pdf', displayName: 'Policy Document' }
];

// Store uploaded files data (keyed by expected file name)
const uploadedFiles = {};

// Initialize file list on page load
document.addEventListener('DOMContentLoaded', () => {
    renderFileList();
});

// Tab switching function
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.file-tab').forEach(tab => {
        tab.classList.remove('active');
    });
    const tabButton = document.querySelector(`[data-tab="${tabName}"]`);
    if (tabButton) {
        tabButton.classList.add('active');
    }
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.style.display = 'none';
        content.classList.remove('active');
    });
    
    // Map tab names to IDs
    const tabIdMap = {
        'files': 'filesTab',
        'summary': 'summaryTab',
        'fact-matrix': 'factMatrixTab',
        'liability-signals': 'liabilitySignalsTab',
        'evidence-completeness': 'evidenceCompletenessTab',
        'coverage-impact': 'coverageImpactTab',
        'liability-recommendation': 'liabilityRecommendationTab',
        'timeline': 'timelineTab',
        'claim-rationale': 'claimRationaleTab',
        'escalation-package': 'escalationPackageTab'
    };
    
    // Load timeline from localStorage when switching to timeline tab
    if (tabName === 'timeline') {
        loadTimeline();
    }
    
    const targetTabId = tabIdMap[tabName];
    if (targetTabId) {
        const targetTab = document.getElementById(targetTabId);
        if (targetTab) {
            targetTab.style.display = 'block';
            targetTab.classList.add('active');
        }
    }
    
    // Hide file content view when switching tabs
    if (tabName === 'files') {
        fileContentView.style.display = 'none';
    }
}

// File input change handler
fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        const expectedFileName = e.target.dataset.expectedFile;
        // Reset input immediately so same file can be selected again
        e.target.value = '';
        handleFileUpload(file, expectedFileName);
    }
});

function renderFileList() {
    fileList.innerHTML = '';
    
    expectedFiles.forEach((expectedFile, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const isUploaded = uploadedFiles[expectedFile.name] !== undefined;
        const fileData = uploadedFiles[expectedFile.name];
        
        fileItem.innerHTML = `
            <div class="file-item-info">
                <span class="file-icon">${getFileIcon(expectedFile.type)}</span>
                <span class="file-name ${isUploaded ? 'uploaded' : ''}" 
                      ${isUploaded ? `onclick="viewFile('${expectedFile.name}')"` : ''}>
                    ${escapeHtml(expectedFile.displayName)}
                </span>
                ${isUploaded ? '<span class="upload-status uploaded">✓ Uploaded</span>' : ''}
            </div>
            ${!isUploaded ? `
                <button class="upload-btn" onclick="triggerFileUpload('${expectedFile.name}', ${index})">
                    Upload
                </button>
            ` : ''}
        `;
        
        fileList.appendChild(fileItem);
    });
}

function getFileIcon(type) {
    if (type === 'image') {
        return '🖼️';
    } else if (type === 'pdf') {
        return '📄';
    }
    return '📎';
}

function triggerFileUpload(expectedFileName, index) {
    // Set the expected file name on the input
    fileInput.dataset.expectedFile = expectedFileName;
    fileInput.click();
}

function handleFileUpload(file, expectedFileName) {
    // Validate file type matches expected file
    const expectedFile = expectedFiles.find(f => f.name === expectedFileName);
    if (!expectedFile) {
        showError('Invalid file selection.');
        return;
    }
    
    // Basic validation - check extension matches
    const fileExt = file.name.toLowerCase().split('.').pop();
    const expectedExt = expectedFileName.toLowerCase().split('.').pop();
    
    if (fileExt !== expectedExt) {
        showError(`Please upload a ${expectedExt.toUpperCase()} file for ${expectedFile.displayName}.`);
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);
    
    // Send to backend
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else {
            // Store uploaded file data
            data.expectedFileName = expectedFileName;
            data.originalFilename = file.name;
            uploadedFiles[expectedFileName] = data;
            
            // Re-render file list to show uploaded status
            renderFileList();
            
            // Show success message briefly
            showSuccess(`${expectedFile.displayName} uploaded successfully!`);
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError(`Failed to upload ${file.name}. Please try again.`);
        console.error('Error:', err);
    });
}

function viewFile(expectedFileName) {
    const fileData = uploadedFiles[expectedFileName];
    if (!fileData) {
        showError('File not found.');
        return;
    }
    
    // Make sure we're on the Files tab
    switchTab('files');
    
    // Hide file list, show content view
    fileListContainer.style.display = 'none';
    fileContentView.style.display = 'block';
    
    // Display file content
    displayContent(fileData);
}

function showFileList() {
    // Switch to Files tab
    switchTab('files');
    
    // Show file list, hide content view
    fileListContainer.style.display = 'block';
    fileContentView.style.display = 'none';
}

function extractFacts() {
    // Check if there are any uploaded files
    const uploadedCount = Object.keys(uploadedFiles).length;
    if (uploadedCount === 0) {
        showError('Please upload at least one file before extracting facts.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare files data to send
    const filesData = Object.values(uploadedFiles);
    
    // Send to backend
    fetch('/extract-facts', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ files: filesData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.facts) {
            // Store facts data
            currentFactsData = data;
            // Display fact matrix
            displayFactMatrix(data);
        } else {
            showError('No facts received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to extract facts. Please try again.');
        console.error('Error:', err);
    });
}

function displayFactMatrix(factsData) {
    // Switch to Fact Matrix tab
    switchTab('fact-matrix');
    
    // Display conflicts if any
    if (factsData.conflicts && factsData.conflicts.length > 0) {
        displayConflicts(factsData.conflicts);
    } else {
        conflictsSection.style.display = 'none';
    }
    
    // Render facts table
    renderFactTable(factsData.facts);
    
    // Scroll to top of tab
    document.getElementById('factMatrixTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderFactTable(facts) {
    factTableBody.innerHTML = '';
    
    if (!facts || facts.length === 0) {
        factTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px;">No facts extracted.</td></tr>';
        return;
    }
    
    facts.forEach((fact, index) => {
        const row = renderFactRow(fact, index);
        factTableBody.appendChild(row);
    });
}

function checkIfFactIsContradicting(fact, factIndex) {
    if (!currentFactsData || !currentFactsData.conflicts || currentFactsData.conflicts.length === 0) {
        return false;
    }
    
    const factSource = fact.source || '';
    const factValue = fact.normalized_value || fact.extracted_fact || '';
    const factValueLower = factValue.toLowerCase().trim();
    
    // Check if this fact is part of any conflict
    for (const conflict of currentFactsData.conflicts) {
        const conflictSources = conflict.sources || [];
        const conflictingValues = conflict.conflicting_values || [];
        
        // Check if fact's source is in conflict sources
        if (conflictSources.includes(factSource)) {
            // Check if fact's value matches any conflicting value
            for (const confValue of conflictingValues) {
                const confValueLower = confValue.toLowerCase().trim();
                // Fuzzy match: check if values are similar
                if (factValueLower === confValueLower || 
                    factValueLower.includes(confValueLower) || 
                    confValueLower.includes(factValueLower)) {
                    return true;
                }
            }
            
            // Also check value_details for more precise matching
            const valueDetails = conflict.value_details || [];
            for (const valueDetail of valueDetails) {
                const detailSources = valueDetail.sources || [];
                const detailValue = valueDetail.value || '';
                const detailValueLower = detailValue.toLowerCase().trim();
                
                if (detailSources.includes(factSource)) {
                    if (factValueLower === detailValueLower || 
                        factValueLower.includes(detailValueLower) || 
                        detailValueLower.includes(factValueLower)) {
                        return true;
                    }
                }
            }
        }
    }
    
    return false;
}

function renderFactRow(fact, index) {
    const row = document.createElement('tr');
    row.dataset.index = index;
    row.dataset.category = fact.category || '';
    row.dataset.source = fact.source || '';
    row.dataset.factText = (fact.extracted_fact || '').toLowerCase();
    
    // Source text (truncated with expand)
    const sourceTextCell = document.createElement('td');
    const sourceText = fact.source_text || '';
    const truncatedText = sourceText.length > 100 ? sourceText.substring(0, 100) + '...' : sourceText;
    sourceTextCell.innerHTML = `
        <div class="source-text-container">
            <span class="source-text-short">${escapeHtml(truncatedText)}</span>
            ${sourceText.length > 100 ? `<button class="expand-text-btn" onclick="toggleSourceText(${index})">Show more</button>` : ''}
            <span class="source-text-full" style="display: none;">${escapeHtml(sourceText)}</span>
        </div>
    `;
    
    // Extracted fact
    const extractedFactCell = document.createElement('td');
    const extractedFactText = fact.extracted_fact || '';
    if (fact.resolved && fact.resolved_value) {
        // Show resolved indicator
        extractedFactCell.innerHTML = `
            <div style="display: flex; align-items: center; gap: 5px;">
                <span>${escapeHtml(extractedFactText)}</span>
                <span class="resolved-badge" title="Resolved from conflict: ${escapeHtml(fact.resolved_value)}" style="background: #4CAF50; color: white; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold;">✓ Resolved</span>
            </div>
        `;
    } else {
        extractedFactCell.textContent = extractedFactText;
    }
    
    // Category (with badge)
    const categoryCell = document.createElement('td');
    const category = fact.category || 'unknown';
    categoryCell.innerHTML = `<span class="category-badge category-${category}">${category}</span>`;
    
    // Source (with badge)
    const sourceCell = document.createElement('td');
    const source = fact.source || 'unknown';
    const sourceIcon = getSourceIcon(source);
    sourceCell.innerHTML = `<span class="source-badge source-${source}">${sourceIcon} ${source.replace('_', ' ')}</span>`;
    
    // Implied (flag indicator)
    const impliedCell = document.createElement('td');
    if (fact.is_implied) {
        const impliedSpan = document.createElement('span');
        impliedSpan.className = 'implied-flag implied-flag-implied';
        impliedSpan.textContent = '⚠️ Implied';
        impliedCell.appendChild(impliedSpan);
    } else {
        const explicitSpan = document.createElement('span');
        explicitSpan.className = 'implied-flag implied-flag-explicit';
        explicitSpan.textContent = '✓ Explicit';
        impliedCell.appendChild(explicitSpan);
    }
    
    // Contradicting (check if fact is part of any conflict)
    const contradictingCell = document.createElement('td');
    const isContradicting = checkIfFactIsContradicting(fact, index);
    if (isContradicting) {
        const contradictingSpan = document.createElement('span');
        contradictingSpan.className = 'contradicting-badge contradicting-yes';
        contradictingSpan.textContent = 'Yes';
        contradictingSpan.style.background = '#f44336';
        contradictingSpan.style.color = 'white';
        contradictingSpan.style.padding = '2px 8px';
        contradictingSpan.style.borderRadius = '3px';
        contradictingSpan.style.fontSize = '12px';
        contradictingSpan.style.fontWeight = 'bold';
        contradictingCell.appendChild(contradictingSpan);
    } else {
        const notContradictingSpan = document.createElement('span');
        notContradictingSpan.className = 'contradicting-badge contradicting-no';
        notContradictingSpan.textContent = 'No';
        notContradictingSpan.style.background = '#4CAF50';
        notContradictingSpan.style.color = 'white';
        notContradictingSpan.style.padding = '2px 8px';
        notContradictingSpan.style.borderRadius = '3px';
        notContradictingSpan.style.fontSize = '12px';
        notContradictingSpan.style.fontWeight = 'bold';
        contradictingCell.appendChild(notContradictingSpan);
    }
    
    // Add resolved class to row if fact is resolved
    if (fact.resolved) {
        row.classList.add('resolved-fact');
        row.style.backgroundColor = '#f0f9f0';
    }
    
    row.appendChild(sourceTextCell);
    row.appendChild(extractedFactCell);
    row.appendChild(categoryCell);
    row.appendChild(sourceCell);
    row.appendChild(impliedCell);
    row.appendChild(contradictingCell);
    
    return row;
}

function getSourceIcon(source) {
    const icons = {
        'fnol': '📋',
        'claimant': '👤',
        'other_driver': '🚗',
        'police': '👮',
        'repair_estimate': '🔧',
        'policy': '🛡️',
        'unknown': '❓'
    };
    return icons[source] || '❓';
}

function toggleSourceText(index) {
    const row = document.querySelector(`tr[data-index="${index}"]`);
    if (!row) return;
    
    const shortText = row.querySelector('.source-text-short');
    const fullText = row.querySelector('.source-text-full');
    const button = row.querySelector('.expand-text-btn');
    
    if (fullText.style.display === 'none') {
        shortText.style.display = 'none';
        fullText.style.display = 'inline';
        button.textContent = 'Show less';
    } else {
        shortText.style.display = 'inline';
        fullText.style.display = 'none';
        button.textContent = 'Show more';
    }
}

function filterFacts() {
    if (!currentFactsData || !currentFactsData.facts) return;
    
    const categoryFilter = document.getElementById('categoryFilter').value;
    const sourceFilter = document.getElementById('sourceFilter').value;
    const searchFilter = document.getElementById('searchFilter').value.toLowerCase();
    const sortBy = document.getElementById('sortBy').value;
    
    let filteredFacts = [...currentFactsData.facts];
    
    // Apply filters
    if (categoryFilter) {
        filteredFacts = filteredFacts.filter(f => f.category === categoryFilter);
    }
    
    if (sourceFilter) {
        filteredFacts = filteredFacts.filter(f => f.source === sourceFilter);
    }
    
    if (searchFilter) {
        filteredFacts = filteredFacts.filter(f => 
            (f.extracted_fact || '').toLowerCase().includes(searchFilter) ||
            (f.source_text || '').toLowerCase().includes(searchFilter) ||
            (f.normalized_value || '').toLowerCase().includes(searchFilter)
        );
    }
    
    // Apply sorting
    if (sortBy === 'confidence') {
        filteredFacts.sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
    } else if (sortBy === 'confidence-asc') {
        filteredFacts.sort((a, b) => (a.confidence || 0) - (b.confidence || 0));
    } else if (sortBy === 'category') {
        filteredFacts.sort((a, b) => (a.category || '').localeCompare(b.category || ''));
    } else if (sortBy === 'source') {
        filteredFacts.sort((a, b) => (a.source || '').localeCompare(b.source || ''));
    }
    
    // Re-render table
    renderFactTable(filteredFacts);
}

function displayConflicts(conflicts) {
    if (!conflicts || conflicts.length === 0) {
        conflictsSection.style.display = 'none';
        return;
    }
    
    conflictsSection.style.display = 'block';
    conflictsContent.innerHTML = '';
    
    conflicts.forEach((conflict, index) => {
        const conflictDiv = document.createElement('div');
        conflictDiv.className = 'conflict-item';
        conflictDiv.setAttribute('data-conflict-index', index);
        
        const recommendedVersion = conflict.recommended_version || '';
        const evidence = conflict.evidence || '';
        const valueDetails = conflict.value_details || [];
        
        // Build value variants display with snippets
        let valueVariantsHtml = '';
        if (valueDetails.length > 0) {
            valueVariantsHtml = valueDetails.map((detail, detailIndex) => {
                const value = detail.value || '';
                const sources = detail.sources || [];
                const snippets = detail.source_snippets || [];
                const isRecommended = value === recommendedVersion;
                
                return `
                    <div class="value-variant ${isRecommended ? 'recommended' : ''}" data-value="${escapeHtml(value)}" data-conflict-index="${index}" data-variant-index="${detailIndex}">
                        <div class="variant-header">
                            <strong>${escapeHtml(value)}</strong>
                            ${isRecommended ? '<span class="recommended-badge">✓ Recommended</span>' : ''}
                        </div>
                        <div class="variant-sources">
                            <strong>Sources:</strong> ${sources.map(s => `<span class="source-badge source-${s}">${getSourceIcon(s)} ${s.replace('_', ' ')}</span>`).join(', ')}
                        </div>
                        ${snippets.length > 0 ? `
                            <div class="variant-snippets">
                                <strong>Source Snippets:</strong>
                                <div class="snippets-list">
                                    ${snippets.map(snippet => `<div class="snippet-item">${escapeHtml(snippet)}</div>`).join('')}
                                </div>
                            </div>
                        ` : ''}
                        <button class="accept-button" onclick="acceptConflictVersion(${index}, ${detailIndex}, '${escapeHtml(value).replace(/'/g, "\\'")}')">
                            ✓ Accept This Version
                        </button>
                    </div>
                `;
            }).join('');
        } else {
            // Fallback to old format if value_details not available
            valueVariantsHtml = `
                <div class="conflict-values-old">
                    <strong>Conflicting Values:</strong>
                    <ul>
                        ${conflict.conflicting_values.map((v, vIndex) => `
                            <li>
                                ${escapeHtml(v)}
                                ${v === recommendedVersion ? ' <span class="recommended-badge">✓ Recommended</span>' : ''}
                                <button class="accept-button-small" onclick="acceptConflictVersion(${index}, ${vIndex}, '${escapeHtml(v).replace(/'/g, "\\'")}')">
                                    Accept
                                </button>
                            </li>
                        `).join('')}
                    </ul>
                </div>
            `;
        }
        
        conflictDiv.innerHTML = `
            <div class="conflict-header">
                <strong>${escapeHtml(conflict.fact_description)}</strong>
                ${conflict.severity ? `<span class="severity-badge severity-${conflict.severity}">${conflict.severity.toUpperCase()}</span>` : ''}
            </div>
            <div class="conflict-details">
                <div class="conflict-sources">
                    <strong>Sources:</strong> ${conflict.sources.map(s => `<span class="source-badge source-${s}">${getSourceIcon(s)} ${s.replace('_', ' ')}</span>`).join(', ')}
                </div>
                ${recommendedVersion && evidence ? `
                    <div class="conflict-recommendation">
                        <div class="recommendation-header">
                            <strong>🤖 AI Recommendation:</strong>
                        </div>
                        <div class="recommended-value">
                            <strong>Recommended Version:</strong> <span class="highlight">${escapeHtml(recommendedVersion)}</span>
                        </div>
                        <div class="recommendation-evidence">
                            <strong>Evidence:</strong>
                            <p>${escapeHtml(evidence)}</p>
                        </div>
                    </div>
                ` : ''}
                <div class="conflict-variants">
                    <strong>Value Variants:</strong>
                    <div class="variants-container">
                        ${valueVariantsHtml}
                    </div>
                </div>
                ${conflict.explanation ? `
                    <div class="conflict-explanation">
                        <strong>Explanation:</strong> ${escapeHtml(conflict.explanation)}
                    </div>
                ` : ''}
            </div>
        `;
        conflictsContent.appendChild(conflictDiv);
    });
}

function acceptConflictVersion(conflictIndex, variantIndex, acceptedValue) {
    if (!currentFactsData || !currentFactsData.conflicts) {
        showError('No conflicts data available.');
        return;
    }
    
    const conflict = currentFactsData.conflicts[conflictIndex];
    if (!conflict) {
        showError('Conflict not found.');
        return;
    }
    
    // Store accepted version
    if (!currentFactsData.acceptedVersions) {
        currentFactsData.acceptedVersions = {};
    }
    currentFactsData.acceptedVersions[conflictIndex] = {
        value: acceptedValue,
        variantIndex: variantIndex,
        timestamp: new Date().toISOString()
    };
    
    // Update fact matrix with accepted value
    if (currentFactsData.facts && conflict.sources && conflict.conflicting_values) {
        const conflictSources = conflict.sources;
        const conflictingValues = conflict.conflicting_values;
        
        // Find the value details for the accepted value
        const valueDetails = conflict.value_details || [];
        const acceptedValueDetail = valueDetails.find(detail => detail.value === acceptedValue);
        const acceptedSources = acceptedValueDetail ? acceptedValueDetail.sources : [];
        const acceptedSnippets = acceptedValueDetail ? acceptedValueDetail.source_snippets || [] : [];
        
        // Find facts that need to be updated (facts that are part of conflict but not in accepted sources)
        const factsToUpdate = [];
        
        currentFactsData.facts.forEach((fact, factIndex) => {
            const factSource = fact.source || '';
            const factValue = fact.normalized_value || fact.extracted_fact || '';
            const factSourceText = fact.source_text || '';
            
            // Check if this fact is part of the conflict sources
            const isInConflictSources = conflictSources.includes(factSource);
            
            if (isInConflictSources) {
                // Check if fact matches any conflicting value
                const matchesConflictingValue = conflictingValues.some(confValue => {
                    const confValueLower = confValue.toLowerCase().trim();
                    const factValueLower = factValue.toLowerCase().trim();
                    // Check if the fact value matches any conflicting value (fuzzy match)
                    return factValueLower === confValueLower || 
                           factValueLower.includes(confValueLower) || 
                           confValueLower.includes(factValueLower);
                });
                
                // Check if fact's source text matches any source snippets from the conflict
                const matchesSourceSnippet = acceptedSnippets.length > 0 ? 
                    acceptedSnippets.some(snippet => {
                        const snippetLower = snippet.toLowerCase();
                        const factSourceTextLower = factSourceText.toLowerCase();
                        // Check if snippet appears in fact's source text (at least 20 chars overlap)
                        return snippetLower.length > 20 && factSourceTextLower.includes(snippetLower.substring(0, Math.min(50, snippetLower.length)));
                    }) : false;
                
                // Determine if this fact should be updated
                // Update if:
                // 1. Fact is in conflict sources AND
                // 2. (Fact doesn't match accepted value OR fact is not in accepted sources) AND
                // 3. (Fact matches a conflicting value OR fact's source text matches conflict snippets)
                const shouldUpdate = isInConflictSources && 
                    (factValue.toLowerCase().trim() !== acceptedValue.toLowerCase().trim()) &&
                    (!acceptedSources.includes(factSource) || matchesSourceSnippet) &&
                    (matchesConflictingValue || matchesSourceSnippet);
                
                if (shouldUpdate) {
                    factsToUpdate.push({ fact, factIndex });
                }
            }
        });
        
        // Update the facts
        factsToUpdate.forEach(({ fact }) => {
            // Update the normalized_value to the accepted value
            fact.normalized_value = acceptedValue;
            
            // Update extracted_fact if it contains conflicting values
            if (fact.extracted_fact) {
                const extractedFactLower = fact.extracted_fact.toLowerCase();
                const conflictingValuesLower = conflictingValues.map(v => v.toLowerCase().trim());
                const factContainsConflictingValue = conflictingValuesLower.some(cv => 
                    extractedFactLower.includes(cv) || cv.includes(extractedFactLower)
                );
                
                if (factContainsConflictingValue) {
                    // Replace conflicting value with accepted value in extracted_fact
                    let updatedExtractedFact = fact.extracted_fact;
                    conflictingValues.forEach(confValue => {
                        const regex = new RegExp(confValue.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
                        updatedExtractedFact = updatedExtractedFact.replace(regex, acceptedValue);
                    });
                    fact.extracted_fact = updatedExtractedFact;
                }
            }
            
            // Mark fact as resolved/updated
            fact.resolved = true;
            fact.resolved_value = acceptedValue;
            fact.resolved_timestamp = new Date().toISOString();
        });
        
        // Re-render the fact table to reflect updates
        if (factsToUpdate.length > 0) {
            renderFactTable(currentFactsData.facts);
        }
    }
    
    // Update UI to show accepted version
    const conflictDiv = document.querySelector(`[data-conflict-index="${conflictIndex}"]`);
    if (conflictDiv) {
        // Remove previous accepted indicators
        conflictDiv.querySelectorAll('.accepted-indicator').forEach(el => el.remove());
        
        // Add accepted indicator
        const variantDiv = conflictDiv.querySelector(`[data-variant-index="${variantIndex}"]`);
        if (variantDiv) {
            const acceptedIndicator = document.createElement('div');
            acceptedIndicator.className = 'accepted-indicator';
            acceptedIndicator.innerHTML = '<span class="accepted-badge">✓ Accepted</span>';
            variantDiv.appendChild(acceptedIndicator);
        }
        
        // Disable accept buttons for this conflict
        conflictDiv.querySelectorAll('.accept-button, .accept-button-small').forEach(btn => {
            btn.disabled = true;
            btn.style.opacity = '0.5';
        });
    }
    
    showSuccess(`Accepted version: "${acceptedValue}" - Fact matrix updated.`);
    
    // Re-analyze liability signals if they exist (to reflect updated facts)
    if (currentLiabilitySignalsData && currentLiabilitySignalsData.signals && currentLiabilitySignalsData.signals.length > 0) {
        showSuccess('Updating liability signals with resolved conflicts...');
        // Automatically re-analyze liability signals with updated facts
        analyzeLiabilitySignals();
    }
}

function exportFactsAsJSON() {
    if (!currentFactsData) {
        showError('No facts data to export.');
        return;
    }
    
    const dataStr = JSON.stringify(currentFactsData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'fact_matrix.json';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function displayContent(data) {
    if (data.type === 'image') {
        // Display image content
        metadataDiv.innerHTML = `
            <h2>Image Information</h2>
            <div class="metadata-item">
                <strong>Filename:</strong> ${escapeHtml(data.filename || data.originalFilename || 'Unknown')}
            </div>
            <div class="metadata-item">
                <strong>Dimensions:</strong> ${data.width} × ${data.height} pixels
            </div>
            <div class="metadata-item">
                <strong>Format:</strong> ${data.format || 'Unknown'}
            </div>
            ${data.size ? `<div class="metadata-item"><strong>Size:</strong> ${formatFileSize(data.size)}</div>` : ''}
        `;
        
        pagesDiv.innerHTML = `
            <div class="image-display">
                <img src="${data.data}" alt="${escapeHtml(data.filename || 'Image')}" class="uploaded-image">
            </div>
        `;
    } else {
        // Display PDF content
        const metadata = data.metadata || {};
        metadataDiv.innerHTML = `
            <h2>Document Information</h2>
            <div class="metadata-item">
                <strong>Filename:</strong> ${escapeHtml(data.filename || data.originalFilename || 'Unknown')}
            </div>
            <div class="metadata-item">
                <strong>Pages:</strong> ${metadata.page_count || 0}
            </div>
            ${metadata.title ? `<div class="metadata-item"><strong>Title:</strong> ${escapeHtml(metadata.title)}</div>` : ''}
            ${metadata.author ? `<div class="metadata-item"><strong>Author:</strong> ${escapeHtml(metadata.author)}</div>` : ''}
            ${metadata.subject ? `<div class="metadata-item"><strong>Subject:</strong> ${escapeHtml(metadata.subject)}</div>` : ''}
        `;
        
        // Display pages
        pagesDiv.innerHTML = '';
        
        if (!data.pages || data.pages.length === 0) {
            pagesDiv.innerHTML = '<div class="empty-text">No content found in PDF.</div>';
        } else {
            data.pages.forEach(page => {
                const pageDiv = document.createElement('div');
                pageDiv.className = 'page';
                
                // Page header
                const pageHeader = document.createElement('div');
                pageHeader.className = 'page-header';
                pageHeader.innerHTML = `
                    <span class="page-number">Page ${page.page_number}</span>
                    <span class="page-dimensions">${Math.round(page.width)} × ${Math.round(page.height)} pts</span>
                `;
                
                // Page text
                const pageText = document.createElement('div');
                pageText.className = 'page-text';
                if (page.text && page.text.trim()) {
                    pageText.textContent = page.text;
                } else {
                    pageText.innerHTML = '<div class="empty-text">No text content on this page.</div>';
                }
                
                // Page images
                const pageImages = document.createElement('div');
                pageImages.className = 'page-images';
                
                if (page.images && page.images.length > 0) {
                    page.images.forEach((img, index) => {
                        const imgDiv = document.createElement('div');
                        imgDiv.className = 'page-image';
                        
                        const imgElement = document.createElement('img');
                        imgElement.src = img.data;
                        imgElement.alt = `Image ${index + 1} from page ${page.page_number}`;
                        imgElement.loading = 'lazy';
                        
                        const imgLabel = document.createElement('div');
                        imgLabel.className = 'page-image-label';
                        imgLabel.textContent = `Image ${index + 1} (${img.ext.toUpperCase()})`;
                        
                        imgDiv.appendChild(imgElement);
                        imgDiv.appendChild(imgLabel);
                        pageImages.appendChild(imgDiv);
                    });
                } else {
                    pageImages.innerHTML = '<div class="empty-images">No images on this page.</div>';
                }
                
                pageDiv.appendChild(pageHeader);
                pageDiv.appendChild(pageText);
                pageDiv.appendChild(pageImages);
                pagesDiv.appendChild(pageDiv);
            });
        }
    }
    
    // Scroll to top of content
    fileContentView.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(message) {
    error.textContent = message;
    error.style.display = 'block';
    loading.style.display = 'none';
    
    setTimeout(() => {
        error.style.display = 'none';
    }, 5000);
}

function showSuccess(message) {
    // Create a temporary success message
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.textContent = message;
    successDiv.style.display = 'block';
    
    // Insert after error div
    error.parentNode.insertBefore(successDiv, error.nextSibling);
    
    setTimeout(() => {
        successDiv.remove();
    }, 3000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function generateSummary() {
    // Check if there are any uploaded files
    const uploadedCount = Object.keys(uploadedFiles).length;
    if (uploadedCount === 0) {
        showError('Please upload at least one file before generating a summary.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare files data to send
    const filesData = Object.values(uploadedFiles);
    
    // Send to backend
    fetch('/generate-summary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ files: filesData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.summary) {
            // Switch to Summary tab and display summary
            switchTab('summary');
            displaySummary(data.summary);
        } else {
            showError('No summary received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to generate summary. Please try again.');
        console.error('Error:', err);
    });
}

function displaySummary(summaryText) {
    // Format and display summary
    let formattedSummary = escapeHtml(summaryText);
    
    // Convert numbered lists (1. 2. 3. etc.)
    formattedSummary = formattedSummary.replace(/(\d+)\.\s+([^\n]+)/g, '<div class="summary-list-item"><strong>$1.</strong> $2</div>');
    
    // Convert bullet points
    formattedSummary = formattedSummary.replace(/^[-•]\s+([^\n]+)/gm, '<div class="summary-list-item">• $1</div>');
    
    // Convert line breaks to HTML
    formattedSummary = formattedSummary.replace(/\n\n+/g, '</p><p>');
    formattedSummary = formattedSummary.replace(/\n/g, '<br>');
    
    // Wrap in paragraphs
    formattedSummary = '<p>' + formattedSummary + '</p>';
    
    // Clean up empty paragraphs
    formattedSummary = formattedSummary.replace(/<p>\s*<\/p>/g, '');
    formattedSummary = formattedSummary.replace(/<p><br><\/p>/g, '');
    
    summaryContent.innerHTML = formattedSummary;
}

function analyzeLiabilitySignals() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before analyzing liability signals.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts data to send
    const factsData = currentFactsData.facts;
    
    // Send to backend
    fetch('/analyze-liability-signals', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.signals) {
            // Store signals data
            currentLiabilitySignalsData = data;
            // Display liability signals
            displayLiabilitySignals(data);
        } else {
            showError('No signals received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to analyze liability signals. Please try again.');
        console.error('Error:', err);
    });
}

function displayLiabilitySignals(signalsData) {
    // Switch to Liability Signals tab
    switchTab('liability-signals');
    
    // Render signals table
    renderLiabilitySignalsTable(signalsData.signals);
    
    // Scroll to top of tab
    document.getElementById('liabilitySignalsTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderLiabilitySignalsTable(signals) {
    const tableBody = document.getElementById('liabilitySignalsTableBody');
    tableBody.innerHTML = '';
    
    if (!signals || signals.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px;">No liability signals identified.</td></tr>';
        return;
    }
    
    signals.forEach((signal, index) => {
        const row = renderLiabilitySignalRow(signal, index);
        tableBody.appendChild(row);
    });
}

function renderLiabilitySignalRow(signal, index) {
    const row = document.createElement('tr');
    row.dataset.index = index;
    
    // Signal Type (with badge)
    const signalTypeCell = document.createElement('td');
    const signalType = signal.signal_type || 'unknown';
    signalTypeCell.innerHTML = `<span class="signal-type-badge signal-type-${signalType.replace(/\s+/g, '-').toLowerCase()}">${escapeHtml(signalType)}</span>`;
    
    // Evidence Text (truncated with expand)
    const evidenceTextCell = document.createElement('td');
    const evidenceText = signal.evidence_text || '';
    const truncatedEvidence = evidenceText.length > 150 ? evidenceText.substring(0, 150) + '...' : evidenceText;
    evidenceTextCell.innerHTML = `
        <div class="evidence-text-container">
            <span class="evidence-text-short">${escapeHtml(truncatedEvidence)}</span>
            ${evidenceText.length > 150 ? `<button class="expand-text-btn" onclick="toggleEvidenceText(${index})">Show more</button>` : ''}
            <span class="evidence-text-full" style="display: none;">${escapeHtml(evidenceText)}</span>
        </div>
    `;
    
    // Impact on Liability
    const impactCell = document.createElement('td');
    impactCell.textContent = signal.impact_on_liability || 'N/A';
    
    // Severity Score (with visual indicator)
    const severityCell = document.createElement('td');
    const severity = signal.severity_score || 0;
    const severityPercent = Math.round(severity * 100);
    severityCell.innerHTML = `
        <div class="severity-container">
            <div class="severity-bar">
                <div class="severity-fill" style="width: ${severityPercent}%"></div>
            </div>
            <span class="severity-text">${severityPercent}%</span>
        </div>
    `;
    
    // Related Facts
    const relatedFactsCell = document.createElement('td');
    const relatedFacts = signal.related_facts || [];
    if (relatedFacts.length > 0) {
        const factsList = relatedFacts.map((fact, idx) => {
            const factText = typeof fact === 'string' ? fact : JSON.stringify(fact);
            const truncatedFact = factText.length > 80 ? factText.substring(0, 80) + '...' : factText;
            return `<div class="related-fact-item">${escapeHtml(truncatedFact)}</div>`;
        }).join('');
        relatedFactsCell.innerHTML = `<div class="related-facts-list">${factsList}</div>`;
    } else {
        relatedFactsCell.textContent = 'None';
    }
    
    // Discrepancies
    const discrepanciesCell = document.createElement('td');
    const discrepancies = signal.discrepancies || '';
    if (discrepancies) {
        discrepanciesCell.innerHTML = `<span class="discrepancy-flag">⚠️ ${escapeHtml(discrepancies)}</span>`;
    } else {
        discrepanciesCell.innerHTML = '<span class="no-discrepancy">✓ None</span>';
    }
    
    row.appendChild(signalTypeCell);
    row.appendChild(evidenceTextCell);
    row.appendChild(impactCell);
    row.appendChild(severityCell);
    row.appendChild(relatedFactsCell);
    row.appendChild(discrepanciesCell);
    
    return row;
}

function toggleEvidenceText(index) {
    const row = document.querySelector(`#liabilitySignalsTableBody tr[data-index="${index}"]`);
    if (!row) return;
    
    const shortText = row.querySelector('.evidence-text-short');
    const fullText = row.querySelector('.evidence-text-full');
    const button = row.querySelector('.expand-text-btn');
    
    if (fullText && button) {
        if (fullText.style.display === 'none') {
            shortText.style.display = 'none';
            fullText.style.display = 'inline';
            button.textContent = 'Show less';
        } else {
            shortText.style.display = 'inline';
            fullText.style.display = 'none';
            button.textContent = 'Show more';
        }
    }
}

function checkEvidenceCompleteness() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before checking evidence completeness.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts data to send
    const factsData = currentFactsData.facts;
    
    // Send to backend
    fetch('/check-evidence-completeness', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.checks || data.missing_evidence) {
            // Display evidence completeness results
            displayEvidenceCompleteness(data);
        } else {
            showError('No completeness data received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to check evidence completeness. Please try again.');
        console.error('Error:', err);
    });
}

function displayEvidenceCompleteness(data) {
    // Switch to Evidence Completeness tab
    switchTab('evidence-completeness');
    
    const checksSummarySection = document.getElementById('checksSummarySection');
    const missingEvidenceSection = document.getElementById('missingEvidenceSection');
    const evidenceEmptyState = document.getElementById('evidenceEmptyState');
    const checksGrid = document.getElementById('checksGrid');
    const missingEvidenceList = document.getElementById('missingEvidenceList');
    
    // Hide empty state
    evidenceEmptyState.style.display = 'none';
    
    // Display checks summary
    if (data.checks) {
        checksSummarySection.style.display = 'block';
        renderChecksSummary(data.checks);
    } else {
        checksSummarySection.style.display = 'none';
    }
    
    // Display missing evidence checklist
    if (data.missing_evidence && data.missing_evidence.length > 0) {
        missingEvidenceSection.style.display = 'block';
        renderMissingEvidence(data.missing_evidence);
    } else {
        missingEvidenceSection.style.display = 'none';
    }
    
    // Scroll to top of tab
    document.getElementById('evidenceCompletenessTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderChecksSummary(checks) {
    const checksGrid = document.getElementById('checksGrid');
    checksGrid.innerHTML = '';
    
    const checkItems = [
        { key: 'turn_by_turn_photos', label: 'Turn-by-Turn Incident Photos', icon: '📸' },
        { key: 'vehicle_damage_angles', label: 'Vehicle Damage Angles', icon: '🚗' },
        { key: 'police_report', label: 'Police Report', icon: '👮' },
        { key: 'timestamps_location', label: 'Timestamps & Location Data', icon: '📍' },
        { key: 'driver_statements', label: 'Driver Statements', icon: '📝' },
        { key: 'document_metadata', label: 'Document Metadata', icon: '📋' }
    ];
    
    checkItems.forEach(item => {
        const check = checks[item.key];
        if (!check) return;
        
        const checkCard = document.createElement('div');
        checkCard.className = 'check-card';
        
        const statusClass = check.status === 'complete' ? 'status-complete' : 
                           check.status === 'partial' ? 'status-partial' : 'status-missing';
        const statusIcon = check.status === 'complete' ? '✅' : 
                         check.status === 'partial' ? '⚠️' : '❌';
        
        checkCard.innerHTML = `
            <div class="check-card-header">
                <span class="check-icon">${item.icon}</span>
                <h4>${escapeHtml(item.label)}</h4>
                <span class="check-status ${statusClass}">${statusIcon} ${check.status || 'unknown'}</span>
            </div>
            <div class="check-card-details">
                <p>${escapeHtml(check.details || 'No details available')}</p>
            </div>
        `;
        
        checksGrid.appendChild(checkCard);
    });
}

function renderMissingEvidence(missingEvidence) {
    const missingEvidenceList = document.getElementById('missingEvidenceList');
    missingEvidenceList.innerHTML = '';
    
    // Sort by priority (high, medium, low)
    const priorityOrder = { 'high': 0, 'medium': 1, 'low': 2 };
    const sortedEvidence = [...missingEvidence].sort((a, b) => {
        const aPriority = priorityOrder[a.priority?.toLowerCase()] ?? 3;
        const bPriority = priorityOrder[b.priority?.toLowerCase()] ?? 3;
        return aPriority - bPriority;
    });
    
    sortedEvidence.forEach((item, index) => {
        const evidenceItem = document.createElement('div');
        evidenceItem.className = 'missing-evidence-item';
        
        const priorityClass = `priority-${item.priority?.toLowerCase() || 'low'}`;
        const priorityIcon = item.priority?.toLowerCase() === 'high' ? '🔴' : 
                           item.priority?.toLowerCase() === 'medium' ? '🟡' : '🟢';
        
        evidenceItem.innerHTML = `
            <div class="missing-evidence-header">
                <span class="evidence-number">${index + 1}</span>
                <h4>${escapeHtml(item.evidence_needed || 'Unknown Evidence')}</h4>
                <span class="priority-badge ${priorityClass}">${priorityIcon} ${item.priority || 'low'}</span>
            </div>
            <div class="missing-evidence-content">
                <div class="why-it-matters">
                    <strong>Why it matters:</strong>
                    <p>${escapeHtml(item.why_it_matters || 'Not specified')}</p>
                </div>
                <div class="suggested-follow-up">
                    <strong>Suggested follow-up questions:</strong>
                    <p>${escapeHtml(item.suggested_follow_up || 'No suggestions provided')}</p>
                </div>
            </div>
        `;
        
        missingEvidenceList.appendChild(evidenceItem);
    });
}

function analyzeCoverageImpact() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before analyzing coverage impact.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts and files data to send
    const factsData = currentFactsData.facts;
    const filesData = Object.values(uploadedFiles);
    
    // Send to backend
    fetch('/analyze-coverage-impact', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData, files: filesData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.coverage_impacts) {
            // Display coverage impact results
            displayCoverageImpact(data);
        } else {
            showError('No coverage impacts received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to analyze coverage impact. Please try again.');
        console.error('Error:', err);
    });
}

function displayCoverageImpact(coverageData) {
    // Switch to Coverage Impact tab
    switchTab('coverage-impact');
    
    // Render coverage impacts table
    renderCoverageImpactTable(coverageData.coverage_impacts);
    
    // Scroll to top of tab
    document.getElementById('coverageImpactTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderCoverageImpactTable(coverageImpacts) {
    const tableBody = document.getElementById('coverageImpactTableBody');
    tableBody.innerHTML = '';
    
    if (!coverageImpacts || coverageImpacts.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px;">No coverage impacts identified.</td></tr>';
        return;
    }
    
    coverageImpacts.forEach((impact, index) => {
        const row = renderCoverageImpactRow(impact, index);
        tableBody.appendChild(row);
    });
}

function renderCoverageImpactRow(impact, index) {
    const row = document.createElement('tr');
    row.dataset.index = index;
    
    // Triggering Fact (truncated with expand)
    const triggeringFactCell = document.createElement('td');
    const triggeringFact = impact.triggering_fact || '';
    const truncatedFact = triggeringFact.length > 150 ? triggeringFact.substring(0, 150) + '...' : triggeringFact;
    triggeringFactCell.innerHTML = `
        <div class="triggering-fact-container">
            <span class="triggering-fact-short">${escapeHtml(truncatedFact)}</span>
            ${triggeringFact.length > 150 ? `<button class="expand-text-btn" onclick="toggleTriggeringFact(${index})">Show more</button>` : ''}
            <span class="triggering-fact-full" style="display: none;">${escapeHtml(triggeringFact)}</span>
        </div>
    `;
    
    // Source (with badge)
    const sourceCell = document.createElement('td');
    const source = impact.source || 'unknown';
    const sourceIcon = getSourceIcon(source);
    sourceCell.innerHTML = `<span class="source-badge source-${source}">${sourceIcon} ${source.replace('_', ' ')}</span>`;
    
    // Policy Clause (truncated with expand)
    const policyClauseCell = document.createElement('td');
    const policyClause = impact.policy_clause || 'Policy document not provided';
    const truncatedClause = policyClause.length > 150 ? policyClause.substring(0, 150) + '...' : policyClause;
    policyClauseCell.innerHTML = `
        <div class="policy-clause-container">
            <span class="policy-clause-short">${escapeHtml(truncatedClause)}</span>
            ${policyClause.length > 150 ? `<button class="expand-text-btn" onclick="togglePolicyClause(${index})">Show more</button>` : ''}
            <span class="policy-clause-full" style="display: none;">${escapeHtml(policyClause)}</span>
        </div>
    `;
    
    // Risk Level (with color-coded badge)
    const riskLevelCell = document.createElement('td');
    const riskLevel = impact.risk_level || 'low';
    const riskClass = `risk-${riskLevel}`;
    const riskIcon = riskLevel === 'high' ? '🔴' : riskLevel === 'medium' ? '🟡' : '🟢';
    riskLevelCell.innerHTML = `<span class="risk-badge ${riskClass}">${riskIcon} ${riskLevel.toUpperCase()}</span>`;
    
    // Coverage Exclusion Type (with badge)
    const exclusionTypeCell = document.createElement('td');
    const exclusionType = impact.coverage_exclusion_type || 'other';
    const exclusionTypeFormatted = exclusionType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    exclusionTypeCell.innerHTML = `<span class="exclusion-type-badge">${escapeHtml(exclusionTypeFormatted)}</span>`;
    
    // Explanation (truncated with expand)
    const explanationCell = document.createElement('td');
    const explanation = impact.explanation || '';
    const truncatedExplanation = explanation.length > 150 ? explanation.substring(0, 150) + '...' : explanation;
    explanationCell.innerHTML = `
        <div class="explanation-container">
            <span class="explanation-short">${escapeHtml(truncatedExplanation)}</span>
            ${explanation.length > 150 ? `<button class="expand-text-btn" onclick="toggleExplanation(${index})">Show more</button>` : ''}
            <span class="explanation-full" style="display: none;">${escapeHtml(explanation)}</span>
            ${impact.evidence ? `<div class="evidence-text" style="margin-top: 5px; font-size: 0.9em; color: #666;"><strong>Evidence:</strong> ${escapeHtml(impact.evidence)}</div>` : ''}
        </div>
    `;
    
    row.appendChild(triggeringFactCell);
    row.appendChild(sourceCell);
    row.appendChild(policyClauseCell);
    row.appendChild(riskLevelCell);
    row.appendChild(exclusionTypeCell);
    row.appendChild(explanationCell);
    
    return row;
}

function toggleTriggeringFact(index) {
    const row = document.querySelector(`#coverageImpactTableBody tr[data-index="${index}"]`);
    if (!row) return;
    
    const shortText = row.querySelector('.triggering-fact-short');
    const fullText = row.querySelector('.triggering-fact-full');
    const button = row.querySelector('.triggering-fact-container .expand-text-btn');
    
    if (fullText && button) {
        if (fullText.style.display === 'none') {
            shortText.style.display = 'none';
            fullText.style.display = 'inline';
            button.textContent = 'Show less';
        } else {
            shortText.style.display = 'inline';
            fullText.style.display = 'none';
            button.textContent = 'Show more';
        }
    }
}

function togglePolicyClause(index) {
    const row = document.querySelector(`#coverageImpactTableBody tr[data-index="${index}"]`);
    if (!row) return;
    
    const shortText = row.querySelector('.policy-clause-short');
    const fullText = row.querySelector('.policy-clause-full');
    const button = row.querySelector('.policy-clause-container .expand-text-btn');
    
    if (fullText && button) {
        if (fullText.style.display === 'none') {
            shortText.style.display = 'none';
            fullText.style.display = 'inline';
            button.textContent = 'Show less';
        } else {
            shortText.style.display = 'inline';
            fullText.style.display = 'none';
            button.textContent = 'Show more';
        }
    }
}

function toggleExplanation(index) {
    const row = document.querySelector(`#coverageImpactTableBody tr[data-index="${index}"]`);
    if (!row) return;
    
    const shortText = row.querySelector('.explanation-short');
    const fullText = row.querySelector('.explanation-full');
    const button = row.querySelector('.explanation-container .expand-text-btn');
    
    if (fullText && button) {
        if (fullText.style.display === 'none') {
            shortText.style.display = 'none';
            fullText.style.display = 'inline';
            button.textContent = 'Show less';
        } else {
            shortText.style.display = 'inline';
            fullText.style.display = 'none';
            button.textContent = 'Show more';
        }
    }
}

function getLiabilityRecommendation() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before getting liability recommendation.');
        return;
    }
    
    // Check if liability signals exist
    if (!currentLiabilitySignalsData || !currentLiabilitySignalsData.signals || currentLiabilitySignalsData.signals.length === 0) {
        showError('Please analyze liability signals first before getting liability recommendation.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts and signals data to send
    const factsData = currentFactsData.facts;
    const signalsData = currentLiabilitySignalsData.signals;
    
    // Send to backend
    fetch('/get-liability-recommendation', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData, signals: signalsData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.claimant_liability_percent !== undefined && data.other_driver_liability_percent !== undefined) {
            // Store recommendation data
            currentLiabilityRecommendationData = data;
            // Display liability recommendation
            displayLiabilityRecommendation(data);
        } else {
            showError('No recommendation received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to get liability recommendation. Please try again.');
        console.error('Error:', err);
    });
}

function displayLiabilityRecommendation(recommendationData) {
    // Switch to Liability Recommendation tab
    switchTab('liability-recommendation');
    
    // Hide empty state, show form
    const emptyState = document.getElementById('recommendationEmptyState');
    const form = document.getElementById('recommendationForm');
    
    if (emptyState) emptyState.style.display = 'none';
    if (form) form.style.display = 'block';
    
    // Populate form fields
    const claimantPercentInput = document.getElementById('claimantLiabilityPercent');
    const otherDriverPercentInput = document.getElementById('otherDriverLiabilityPercent');
    const explanationTextarea = document.getElementById('recommendationExplanation');
    const keyFactorsList = document.getElementById('keyFactorsList');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    
    if (claimantPercentInput) {
        claimantPercentInput.value = recommendationData.claimant_liability_percent || 0;
    }
    
    if (otherDriverPercentInput) {
        otherDriverPercentInput.value = recommendationData.other_driver_liability_percent || 0;
    }
    
    if (explanationTextarea) {
        explanationTextarea.value = recommendationData.explanation || '';
    }
    
    // Render key factors
    if (keyFactorsList) {
        keyFactorsList.innerHTML = '';
        const keyFactors = recommendationData.key_factors || [];
        if (keyFactors.length > 0) {
            keyFactors.forEach(factor => {
                const li = document.createElement('li');
                li.textContent = factor;
                keyFactorsList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No key factors identified.';
            li.style.color = '#999';
            keyFactorsList.appendChild(li);
        }
    }
    
    // Update confidence display
    const confidence = recommendationData.confidence || 0;
    const confidencePercent = Math.round(confidence * 100);
    if (confidenceFill) {
        confidenceFill.style.width = confidencePercent + '%';
    }
    if (confidenceText) {
        confidenceText.textContent = confidencePercent + '%';
    }
    
    // Scroll to top of tab
    document.getElementById('liabilityRecommendationTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateLiabilityPercentages() {
    const claimantPercentInput = document.getElementById('claimantLiabilityPercent');
    const otherDriverPercentInput = document.getElementById('otherDriverLiabilityPercent');
    
    if (!claimantPercentInput || !otherDriverPercentInput) return;
    
    let claimantPercent = parseInt(claimantPercentInput.value) || 0;
    let otherDriverPercent = parseInt(otherDriverPercentInput.value) || 0;
    
    // Ensure values are within valid range
    claimantPercent = Math.max(0, Math.min(100, claimantPercent));
    otherDriverPercent = Math.max(0, Math.min(100, otherDriverPercent));
    
    // Normalize to sum to 100
    const total = claimantPercent + otherDriverPercent;
    if (total !== 100 && total > 0) {
        claimantPercent = Math.round((claimantPercent / total) * 100);
        otherDriverPercent = 100 - claimantPercent;
    } else if (total === 0) {
        claimantPercent = 50;
        otherDriverPercent = 50;
    }
    
    // Update input values
    claimantPercentInput.value = claimantPercent;
    otherDriverPercentInput.value = otherDriverPercent;
}

function updateLiabilityRecommendation() {
    if (!currentLiabilityRecommendationData) {
        showError('No recommendation data to update.');
        return;
    }
    
    const claimantPercentInput = document.getElementById('claimantLiabilityPercent');
    const otherDriverPercentInput = document.getElementById('otherDriverLiabilityPercent');
    const explanationTextarea = document.getElementById('recommendationExplanation');
    
    if (!claimantPercentInput || !otherDriverPercentInput || !explanationTextarea) {
        showError('Form fields not found.');
        return;
    }
    
    // Update stored data with edited values
    currentLiabilityRecommendationData.claimant_liability_percent = parseInt(claimantPercentInput.value) || 0;
    currentLiabilityRecommendationData.other_driver_liability_percent = parseInt(otherDriverPercentInput.value) || 0;
    currentLiabilityRecommendationData.explanation = explanationTextarea.value || '';
    
    showSuccess('Liability recommendation updated successfully!');
}

function generateTimeline() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before generating timeline.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts data to send
    const factsData = currentFactsData.facts;
    
    // Send to backend
    fetch('/generate-timeline', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.timeline) {
            // Store timeline data
            currentTimelineData = data;
            // Display timeline
            displayTimeline(data);
        } else {
            showError('No timeline received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to generate timeline. Please try again.');
        console.error('Error:', err);
    });
}

function displayTimeline(timelineData) {
    // Switch to Timeline tab
    switchTab('timeline');
    
    // Hide empty state, show timeline container
    const emptyState = document.getElementById('timelineEmptyState');
    const timelineContainer = document.getElementById('timelineEventsContainer');
    const saveButton = document.getElementById('saveTimelineButton');
    
    if (emptyState) emptyState.style.display = 'none';
    if (timelineContainer) timelineContainer.style.display = 'block';
    if (saveButton) saveButton.style.display = 'inline-flex';
    
    // Render timeline events
    renderTimelineEvents(timelineData.timeline);
    
    // Scroll to top of tab
    document.getElementById('timelineTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderTimelineEvents(timeline) {
    const timelineList = document.getElementById('timelineEventsList');
    if (!timelineList) return;
    
    timelineList.innerHTML = '';
    
    if (!timeline || timeline.length === 0) {
        timelineList.innerHTML = '<li class="empty-timeline">No events in timeline.</li>';
        return;
    }
    
    timeline.forEach((event, index) => {
        const eventItem = renderTimelineEvent(event, index);
        timelineList.appendChild(eventItem);
    });
}

function renderTimelineEvent(event, index) {
    const li = document.createElement('li');
    li.className = 'timeline-event';
    li.dataset.index = index;
    
    const eventNumber = event.event_number || (index + 1);
    const description = event.description || '';
    const timestamp = event.timestamp || '';
    const supportingFacts = event.supporting_facts || [];
    
    li.innerHTML = `
        <div class="event-header">
            <span class="event-number">Event ${eventNumber}</span>
            ${timestamp ? `<span class="event-timestamp">${escapeHtml(timestamp)}</span>` : ''}
            <div class="event-controls">
                <button class="event-control-btn" onclick="reorderEvent(${index}, 'up')" ${index === 0 ? 'disabled' : ''} title="Move up">↑</button>
                <button class="event-control-btn" onclick="reorderEvent(${index}, 'down')" ${index === (currentTimelineData?.timeline?.length - 1 || 0) ? 'disabled' : ''} title="Move down">↓</button>
            </div>
        </div>
        <div class="event-description-container">
            <textarea class="event-description" rows="3" onchange="updateEventDescription(${index}, this.value)">${escapeHtml(description)}</textarea>
        </div>
        ${supportingFacts.length > 0 ? `
            <div class="supporting-facts">
                <button class="supporting-facts-toggle" onclick="toggleSupportingFacts(${index})">
                    <span class="toggle-icon">▼</span>
                    <span>Supporting Facts (${supportingFacts.length})</span>
                </button>
                <div class="supporting-facts-content" style="display: none;">
                    <ul class="supporting-facts-list">
                        ${supportingFacts.map(fact => {
                            // Parse fact number from format "Fact 1", "Fact 5", etc.
                            const factMatch = fact.match(/Fact\s+(\d+)/i);
                            if (factMatch) {
                                const factIndex = parseInt(factMatch[1]) - 1; // Convert to 0-based index
                                return `<li><a href="#" onclick="scrollToFact(${factIndex}); return false;" class="fact-link" style="color: #667eea; text-decoration: underline; cursor: pointer;">${escapeHtml(fact)}</a></li>`;
                            } else {
                                // Backward compatibility: show as plain text if not in "Fact X" format
                                return `<li>${escapeHtml(fact)}</li>`;
                            }
                        }).join('')}
                    </ul>
                </div>
            </div>
        ` : ''}
    `;
    
    return li;
}

function updateEventDescription(index, newDescription) {
    if (!currentTimelineData || !currentTimelineData.timeline) {
        return;
    }
    
    if (index >= 0 && index < currentTimelineData.timeline.length) {
        currentTimelineData.timeline[index].description = newDescription;
        currentTimelineData.timeline[index].edited = true;
    }
}

function reorderEvent(index, direction) {
    if (!currentTimelineData || !currentTimelineData.timeline) {
        return;
    }
    
    const timeline = currentTimelineData.timeline;
    if (index < 0 || index >= timeline.length) {
        return;
    }
    
    let newIndex;
    if (direction === 'up' && index > 0) {
        newIndex = index - 1;
    } else if (direction === 'down' && index < timeline.length - 1) {
        newIndex = index + 1;
    } else {
        return;
    }
    
    // Swap events
    [timeline[index], timeline[newIndex]] = [timeline[newIndex], timeline[index]];
    
    // Update event numbers
    timeline.forEach((event, idx) => {
        event.event_number = idx + 1;
    });
    
    // Re-render timeline
    renderTimelineEvents(timeline);
}

function toggleSupportingFacts(index) {
    const eventItem = document.querySelector(`.timeline-event[data-index="${index}"]`);
    if (!eventItem) return;
    
    const content = eventItem.querySelector('.supporting-facts-content');
    const toggleIcon = eventItem.querySelector('.toggle-icon');
    
    if (content && toggleIcon) {
        if (content.style.display === 'none') {
            content.style.display = 'block';
            toggleIcon.textContent = '▲';
        } else {
            content.style.display = 'none';
            toggleIcon.textContent = '▼';
        }
    }
}

function scrollToFact(factIndex) {
    // Switch to fact matrix tab
    switchTab('fact-matrix');
    
    // Wait for tab to be visible, then scroll to the fact
    setTimeout(() => {
        const factRow = document.querySelector(`tr[data-index="${factIndex}"]`);
        if (factRow) {
            // Highlight the row temporarily
            factRow.style.backgroundColor = '#fff3cd';
            factRow.scrollIntoView({ behavior: 'smooth', block: 'center' });
            
            // Remove highlight after 2 seconds
            setTimeout(() => {
                factRow.style.backgroundColor = '';
            }, 2000);
        } else {
            showError(`Fact ${factIndex + 1} not found in fact matrix.`);
        }
    }, 100);
}

function saveTimeline() {
    if (!currentTimelineData) {
        showError('No timeline data to save.');
        return;
    }
    
    try {
        localStorage.setItem('timeline_reconstruction', JSON.stringify(currentTimelineData));
        showSuccess('Timeline saved successfully!');
    } catch (err) {
        showError('Failed to save timeline. Please try again.');
        console.error('Error saving timeline:', err);
    }
}

function loadTimeline() {
    try {
        const savedTimeline = localStorage.getItem('timeline_reconstruction');
        if (savedTimeline) {
            const timelineData = JSON.parse(savedTimeline);
            currentTimelineData = timelineData;
            displayTimeline(timelineData);
        }
    } catch (err) {
        console.error('Error loading timeline:', err);
    }
}

function generateClaimRationale() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before generating claim rationale.');
        return;
    }
    
    // Check if liability signals exist
    if (!currentLiabilitySignalsData || !currentLiabilitySignalsData.signals || currentLiabilitySignalsData.signals.length === 0) {
        showError('Please analyze liability signals first before generating claim rationale.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts, signals, and files data to send
    const factsData = currentFactsData.facts;
    const signalsData = currentLiabilitySignalsData.signals;
    const filesData = Object.values(uploadedFiles);
    
    // Send to backend
    fetch('/generate-claim-rationale', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData, signals: signalsData, files: filesData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.rationale) {
            // Store rationale data
            currentClaimRationaleData = data;
            // Display rationale
            displayClaimRationale(data.rationale);
        } else {
            showError('No rationale received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to generate claim rationale. Please try again.');
        console.error('Error:', err);
    });
}

function displayClaimRationale(rationale) {
    // Switch to Claim Rationale tab
    switchTab('claim-rationale');
    
    // Hide empty state, show display
    const emptyState = document.getElementById('rationaleEmptyState');
    const display = document.getElementById('rationaleDisplay');
    
    if (emptyState) emptyState.style.display = 'none';
    if (display) display.style.display = 'block';
    
    // Render rationale
    renderClaimRationale(rationale);
    
    // Scroll to top of tab
    document.getElementById('claimRationaleTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderClaimRationale(rationale) {
    const display = document.getElementById('rationaleDisplay');
    if (!display) return;
    
    let html = '<div class="rationale-sections">';
    
    // Incident Summary
    if (rationale.incident_summary) {
        html += `
            <div class="rationale-section">
                <h3>Incident Summary</h3>
                <div class="rationale-content">${escapeHtml(rationale.incident_summary).replace(/\n/g, '<br>')}</div>
            </div>
        `;
    }
    
    // Evidence Overview
    if (rationale.evidence_overview) {
        html += `
            <div class="rationale-section">
                <h3>Evidence Overview</h3>
                <div class="rationale-content">
                    ${rationale.evidence_overview.narratives ? `
                        <div class="evidence-subsection">
                            <strong>Narratives:</strong>
                            <p>${escapeHtml(rationale.evidence_overview.narratives).replace(/\n/g, '<br>')}</p>
                        </div>
                    ` : ''}
                    ${rationale.evidence_overview.photos ? `
                        <div class="evidence-subsection">
                            <strong>Photos:</strong>
                            <p>${escapeHtml(rationale.evidence_overview.photos).replace(/\n/g, '<br>')}</p>
                        </div>
                    ` : ''}
                    ${rationale.images && rationale.images.length > 0 ? `
                        <div class="evidence-subsection">
                            <strong>Uploaded Images:</strong>
                            <div class="rationale-images-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                                ${rationale.images.map((img, imgIndex) => {
                                    const sourceLabel = img.type === 'pdf_image' 
                                        ? `${escapeHtml(img.source)} - Page ${img.page}`
                                        : escapeHtml(img.source);
                                    return `
                                        <div class="rationale-image-item" style="border: 1px solid #ddd; border-radius: 5px; padding: 10px; background: #f9f9f9;">
                                            <img src="${img.data}" alt="Evidence image ${imgIndex + 1}" 
                                                 style="width: 100%; height: auto; border-radius: 3px; cursor: pointer;"
                                                 onclick="window.open('${img.data}', '_blank')"
                                                 onerror="this.style.display='none'">
                                            <div style="margin-top: 8px; font-size: 12px; color: #666; text-align: center;">
                                                ${sourceLabel}
                                            </div>
                                        </div>
                                    `;
                                }).join('')}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
    }
    
    // Liability Assessment Logic
    if (rationale.liability_assessment_logic) {
        html += `
            <div class="rationale-section">
                <h3>Liability Assessment Logic</h3>
                <div class="rationale-content">${escapeHtml(rationale.liability_assessment_logic).replace(/\n/g, '<br>')}</div>
            </div>
        `;
    }
    
    // Key Evidence
    if (rationale.key_evidence && rationale.key_evidence.length > 0) {
        html += `
            <div class="rationale-section">
                <h3>Key Evidence Supporting Assessment</h3>
                <ul class="rationale-list">
                    ${rationale.key_evidence.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Open Questions
    if (rationale.open_questions && rationale.open_questions.length > 0) {
        html += `
            <div class="rationale-section">
                <h3>Open Questions / Follow-Up</h3>
                <ul class="rationale-list">
                    ${rationale.open_questions.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Coverage Considerations
    if (rationale.coverage_considerations) {
        html += `
            <div class="rationale-section">
                <h3>Coverage Considerations</h3>
                <div class="rationale-content">${escapeHtml(rationale.coverage_considerations).replace(/\n/g, '<br>')}</div>
            </div>
        `;
    }
    
    // Recommendation
    if (rationale.recommendation) {
        html += `
            <div class="rationale-section">
                <h3>Recommendation</h3>
                <div class="rationale-content recommendation-highlight">${escapeHtml(rationale.recommendation).replace(/\n/g, '<br>')}</div>
            </div>
        `;
    }
    
    html += '</div>';
    display.innerHTML = html;
}

function generateEscalationPackage() {
    // Check if fact matrix exists
    if (!currentFactsData || !currentFactsData.facts || currentFactsData.facts.length === 0) {
        showError('Please extract facts first before generating escalation package.');
        return;
    }
    
    // Check if liability signals exist
    if (!currentLiabilitySignalsData || !currentLiabilitySignalsData.signals || currentLiabilitySignalsData.signals.length === 0) {
        showError('Please analyze liability signals first before generating escalation package.');
        return;
    }
    
    // Show loading state
    loading.style.display = 'block';
    error.style.display = 'none';
    
    // Prepare facts and signals data to send
    const factsData = currentFactsData.facts;
    const signalsData = currentLiabilitySignalsData.signals;
    
    // Send to backend
    fetch('/generate-escalation-package', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ facts: factsData, signals: signalsData })
    })
    .then(response => response.json())
    .then(data => {
        loading.style.display = 'none';
        
        if (data.error) {
            showError(data.error);
        } else if (data.escalation_package) {
            // Store escalation package data
            currentEscalationPackageData = data;
            // Display escalation package
            displayEscalationPackage(data.escalation_package);
        } else {
            showError('No escalation package received from server.');
        }
    })
    .catch(err => {
        loading.style.display = 'none';
        showError('Failed to generate escalation package. Please try again.');
        console.error('Error:', err);
    });
}

function displayEscalationPackage(escalationPackage) {
    // Switch to Escalation Package tab
    switchTab('escalation-package');
    
    // Hide empty state, show display
    const emptyState = document.getElementById('escalationEmptyState');
    const display = document.getElementById('escalationDisplay');
    
    if (emptyState) emptyState.style.display = 'none';
    if (display) display.style.display = 'block';
    
    // Render escalation package
    renderEscalationPackage(escalationPackage);
    
    // Scroll to top of tab
    document.getElementById('escalationPackageTab').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderEscalationPackage(escalationPackage) {
    const display = document.getElementById('escalationDisplay');
    if (!display) return;
    
    let html = '<div class="escalation-sections">';
    
    // Executive Summary
    if (escalationPackage.executive_summary) {
        html += `
            <div class="escalation-section executive-summary-section">
                <h3>Executive Summary</h3>
                <div class="escalation-content">${escapeHtml(escalationPackage.executive_summary).replace(/\n/g, '<br>')}</div>
            </div>
        `;
    }
    
    // Top 5 Risks
    if (escalationPackage.top_5_risks && escalationPackage.top_5_risks.length > 0) {
        html += `
            <div class="escalation-section">
                <h3>Top 5 Risks</h3>
                <div class="risks-list">
                    ${escalationPackage.top_5_risks.map((risk, index) => {
                        const severityClass = `risk-severity-${risk.severity || 'medium'}`;
                        const severityIcon = risk.severity === 'high' ? '🔴' : risk.severity === 'medium' ? '🟡' : '🟢';
                        return `
                            <div class="risk-item">
                                <div class="risk-header">
                                    <span class="risk-number">${index + 1}</span>
                                    <span class="risk-severity ${severityClass}">${severityIcon} ${(risk.severity || 'medium').toUpperCase()}</span>
                                </div>
                                <div class="risk-content">
                                    <strong>Risk:</strong> ${escapeHtml(risk.risk || 'N/A')}
                                </div>
                                ${risk.impact ? `
                                    <div class="risk-impact">
                                        <strong>Impact:</strong> ${escapeHtml(risk.impact)}
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        `;
    }
    
    // Needed Supervisor Decisions
    if (escalationPackage.needed_supervisor_decisions && escalationPackage.needed_supervisor_decisions.length > 0) {
        html += `
            <div class="escalation-section">
                <h3>Needed Supervisor Decisions</h3>
                <ul class="escalation-list">
                    ${escalationPackage.needed_supervisor_decisions.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    // Recommended Adjuster Actions
    if (escalationPackage.recommended_adjuster_actions && escalationPackage.recommended_adjuster_actions.length > 0) {
        html += `
            <div class="escalation-section">
                <h3>Recommended Adjuster Actions</h3>
                <ul class="escalation-list">
                    ${escalationPackage.recommended_adjuster_actions.map(item => `<li>${escapeHtml(item)}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    html += '</div>';
    display.innerHTML = html;
}

// Make functions globally accessible
window.triggerFileUpload = triggerFileUpload;
window.viewFile = viewFile;
window.extractFacts = extractFacts;
window.filterFacts = filterFacts;
window.toggleSourceText = toggleSourceText;
window.exportFactsAsJSON = exportFactsAsJSON;
window.switchTab = switchTab;
window.generateSummary = generateSummary;
window.analyzeLiabilitySignals = analyzeLiabilitySignals;
window.toggleEvidenceText = toggleEvidenceText;
window.checkEvidenceCompleteness = checkEvidenceCompleteness;
window.analyzeCoverageImpact = analyzeCoverageImpact;
window.toggleTriggeringFact = toggleTriggeringFact;
window.togglePolicyClause = togglePolicyClause;
window.toggleExplanation = toggleExplanation;
window.getLiabilityRecommendation = getLiabilityRecommendation;
window.updateLiabilityPercentages = updateLiabilityPercentages;
window.updateLiabilityRecommendation = updateLiabilityRecommendation;
window.generateTimeline = generateTimeline;
window.updateEventDescription = updateEventDescription;
window.reorderEvent = reorderEvent;
window.toggleSupportingFacts = toggleSupportingFacts;
window.saveTimeline = saveTimeline;
window.scrollToFact = scrollToFact;
window.generateClaimRationale = generateClaimRationale;
window.generateEscalationPackage = generateEscalationPackage;
window.acceptConflictVersion = acceptConflictVersion;
