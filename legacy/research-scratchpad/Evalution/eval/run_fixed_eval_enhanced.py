import os
import sys
import subprocess
import pandas as pd
import requests
import time

def check_servers():
    """Check if all fusion servers are running"""
    servers = {
        "Model Server": "http://localhost:9000/health",
        "Early Fusion": "http://localhost:8001/health", 
        "Late Fusion": "http://localhost:8002/health",
        "Hybrid Fusion": "http://localhost:8003/health"
    }
    
    print("Checking server status...")
    all_running = True
    
    for name, url in servers.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: Running")
            else:
                print(f"❌ {name}: Not responding properly")
                all_running = False
        except Exception as e:
            print(f"❌ {name}: Not running - {e}")
            all_running = False
    
    return all_running

def check_document():
    """Check if the document exists and return absolute path"""
    # Try multiple possible locations
    possible_paths = [
        "../documents/xyz.pdf",
        "s:/Multimodal Project/Evalution/documents/xyz.pdf",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents", "xyz.pdf")
    ]
    
    for doc_path in possible_paths:
        abs_path = os.path.abspath(doc_path)
        if os.path.exists(abs_path):
            print(f"✅ Document found: {abs_path}")
            return abs_path
    
    print(f"❌ Document not found in any of these locations:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    
    # Show what's actually in the documents directory
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")
    if os.path.exists(docs_dir):
        print(f"\nAvailable files in {docs_dir}:")
        for file in os.listdir(docs_dir):
            print(f"  - {file}")
    return None

def test_single_query():
    """Test a single query to see what's happening"""
    print("\nTesting single query...")
    
    # Get absolute document path
    doc_path = check_document()
    if not doc_path:
        print("Cannot test - document not found")
        return
    
    test_query = {
        "question": "How many total orders were delivered?",
        "documents": [doc_path]  # Use absolute path and documents array
    }
    
    servers = {
        "Early Fusion": "http://localhost:8001/ask",
        "Late Fusion": "http://localhost:8002/ask", 
        "Hybrid Fusion": "http://localhost:8003/ask"
    }
    
    for name, endpoint in servers.items():
        try:
            print(f"\nTesting {name}...")
            response = requests.post(
                endpoint, 
                json=test_query,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'error' in result:
                    print(f"  Error: {result['error']}")
                else:
                    print(f"  Answer: {result.get('answer', 'No answer')}")
                    print(f"  Confidence: {result.get('confidence', 'No confidence')}")
                    print(f"  Latency: {result.get('latency_ms', 'No latency')} ms")
            else:
                print(f"  HTTP Error: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"  Error calling {name}: {e}")

def run_evaluation():
    """Run the actual evaluation"""
    print("\n" + "="*50)
    print("RUNNING FULL EVALUATION")
    print("="*50)
    
    # Make sure we're in the right directory
    original_dir = os.getcwd()
    eval_dir = "s:/Multimodal Project/Evalution/eval"
    os.chdir(eval_dir)
    
    # Ensure document path is absolute for the evaluation
    doc_path = check_document()
    if not doc_path:
        print("Cannot run evaluation - document not found")
        return False
    
    # Update queries file to use absolute path if needed
    queries_file = "queries_small.csv"
    if os.path.exists(queries_file):
        try:
            import pandas as pd
            queries_df = pd.read_csv(queries_file)
            
            # Update doc_path column to use absolute path
            if 'doc_path' in queries_df.columns:
                queries_df['doc_path'] = doc_path
                queries_df.to_csv(queries_file, index=False)
                print(f"Updated queries file to use absolute path: {doc_path}")
        except Exception as e:
            print(f"Warning: Could not update queries file: {e}")
    
    # Run the evaluation
    try:
        result = subprocess.run([
            "python", "run_eval.py",
            "--queries", queries_file,
            "--out", "results_small.csv", 
            "--summary", "summary_small.csv"
        ], capture_output=True, text=True, timeout=600)  # Increased timeout to 10 minutes
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        if result.returncode == 0:
            print("\n✅ Evaluation completed successfully!")
            return True
        else:
            print(f"\n❌ Evaluation failed with return code: {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Evaluation timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running evaluation: {e}")
        return False
    finally:
        os.chdir(original_dir)

def analyze_results():
    """Analyze the results and show comprehensive accuracy comparison"""
    try:
        # Load results
        results = pd.read_csv("results_small.csv")
        summary = pd.read_csv("summary_small.csv")
        
        print("\n" + "="*60)
        print("COMPREHENSIVE ACCURACY EVALUATION WITH ENHANCED METRICS")
        print("="*60)
        
        # Enhanced summary with all accuracy metrics
        print("\nPERFORMANCE METRICS BY SYSTEM:")
        print("-" * 90)
        
        # Display metrics including new ones
        metrics_cols = ['system', 'confidence', 'latency_ms', 'EM', 'F1', 'ROUGE_L', 'METEOR', 'BLEU', 'Num_Acc@tol', 'Num_MAE']
        available_cols = [col for col in metrics_cols if col in summary.columns]
        summary_display = summary[available_cols].copy()
        
        # Format for better readability
        for col in ['confidence', 'EM', 'F1', 'ROUGE_L', 'METEOR', 'BLEU', 'Num_Acc@tol']:
            if col in summary_display.columns:
                summary_display[col] = summary_display[col].round(4)
        
        if 'latency_ms' in summary_display.columns:
            summary_display['latency_ms'] = summary_display['latency_ms'].round(1)
        if 'Num_MAE' in summary_display.columns:
            summary_display['Num_MAE'] = summary_display['Num_MAE'].round(3)
            
        print(summary_display.to_string(index=False))
        
        # Check if all models performed identically
        identical_performance = True
        if len(summary) > 1:
            first_row = summary.iloc[0]
            for _, row in summary.iloc[1:].iterrows():
                for col in ['EM', 'F1', 'ROUGE_L', 'METEOR', 'BLEU', 'Num_Acc@tol', 'confidence']:
                    if col in summary.columns:
                        if abs(first_row[col] - row[col]) > 1e-6:
                            identical_performance = False
                            break
                if not identical_performance:
                    break
        
        if identical_performance:
            print(f"\n⚠️  IDENTICAL PERFORMANCE DETECTED")
            print("All fusion models produced identical results.")
            print("This suggests:")
            print("• Models may be using similar fusion strategies")
            print("• Document content may not benefit from different fusion approaches")
            print("• Small dataset size (only 3 questions) limits differentiation")
            print("• Models may need different hyperparameter tuning")
        
        # Detailed accuracy breakdown
        print(f"\n" + "="*60)
        print("DETAILED ACCURACY ANALYSIS")
        print("="*60)
        
        for system in summary['system'].unique():
            system_results = results[results['system'] == system]
            system_summary = summary[summary['system'] == system].iloc[0]
            
            print(f"\n🔍 {system.upper()}:")
            print(f"   Total Questions: {len(system_results)}")
            print(f"   Exact Match (EM): {system_summary.get('EM', 0):.1%}")
            print(f"   F1 Score: {system_summary.get('F1', 0):.4f}")
            
            # New metrics
            if 'ROUGE_L' in system_summary:
                print(f"   ROUGE-L Score: {system_summary['ROUGE_L']:.4f}")
            if 'METEOR' in system_summary:
                print(f"   METEOR Score: {system_summary['METEOR']:.4f}")
            if 'BLEU' in system_summary:
                print(f"   BLEU Score: {system_summary['BLEU']:.4f}")
                
            print(f"   Numeric Accuracy: {system_summary.get('Num_Acc@tol', 0):.1%}")
            print(f"   Mean Absolute Error: {system_summary.get('Num_MAE', 0):.3f}")
            print(f"   Average Confidence: {system_summary.get('confidence', 0):.4f}")
            print(f"   Average Latency: {system_summary.get('latency_ms', 0):.1f} ms")
            
            # Error analysis
            errors = system_results[system_results['error'].notna()]
            if not errors.empty:
                print(f"   ⚠️  Errors: {len(errors)}")
            
            # Question type breakdown
            numeric_questions = system_results[system_results['gold_num_norm'].notna()]
            text_questions = system_results[system_results['gold_num_norm'].isna()]
            
            if len(numeric_questions) > 0:
                numeric_acc = (numeric_questions['Num_Acc@tol'] == 1.0).mean()
                print(f"   Numeric Questions ({len(numeric_questions)}): {numeric_acc:.1%} accuracy")
                
                # Show which numeric questions failed
                failed_numeric = numeric_questions[numeric_questions['Num_Acc@tol'] != 1.0]
                if len(failed_numeric) > 0:
                    print(f"   Failed numeric questions:")
                    for _, row in failed_numeric.iterrows():
                        print(f"     • Q: {row['question'][:50]}...")
                        print(f"       Expected: {row['gold_num_norm']}, Got: {row['pred_num_norm']}")
            
            if len(text_questions) > 0:
                text_em = text_questions['EM'].mean()
                print(f"   Text Questions ({len(text_questions)}): {text_em:.1%} exact match")
                
                # Show text quality metrics if available
                if 'ROUGE_L' in text_questions.columns:
                    rouge_avg = text_questions['ROUGE_L'].mean()
                    print(f"   Text ROUGE-L Average: {rouge_avg:.4f}")
                if 'METEOR' in text_questions.columns:
                    meteor_avg = text_questions['METEOR'].mean()
                    print(f"   Text METEOR Average: {meteor_avg:.4f}")
                if 'BLEU' in text_questions.columns:
                    bleu_avg = text_questions['BLEU'].mean()
                    print(f"   Text BLEU Average: {bleu_avg:.4f}")
        
        # Model ranking and selection
        print(f"\n" + "="*60)
        print("MODEL RANKING & SELECTION (ENHANCED)")
        print("="*60)
        
        if identical_performance:
            print("\n🤝 PERFORMANCE TIE:")
            print("All models performed identically. Selection based on latency:")
            
            # Sort by latency (lower is better) when performance is identical
            ranking = summary.sort_values('latency_ms', ascending=True)
            
            for i, (_, row) in enumerate(ranking.iterrows(), 1):
                print(f"{i}. {row['system']} - {row['latency_ms']:.1f} ms")
            
            best_system = ranking.iloc[0]['system']
            print(f"\n🏆 RECOMMENDED MODEL: {best_system.upper()}")
            print(f"📊 Reason: Fastest among identical performers ({ranking.iloc[0]['latency_ms']:.1f} ms)")
            
        else:
            # Calculate composite score (weighted combination of metrics including new ones)
            summary_scored = summary.copy()
            
            # Safer normalization - handle zero variance
            score_metrics = ['EM', 'F1', 'ROUGE_L', 'METEOR', 'BLEU', 'Num_Acc@tol', 'confidence']
            for col in score_metrics:
                if col in summary_scored.columns:
                    col_max = summary_scored[col].max()
                    if col_max > 0:
                        summary_scored[f'norm_{col.lower()}'] = summary_scored[col] / col_max
                    else:
                        summary_scored[f'norm_{col.lower()}'] = 0.0
            
            # Inverse normalize latency and MAE (lower is better)
            if summary_scored['latency_ms'].max() > summary_scored['latency_ms'].min():
                summary_scored['norm_latency'] = summary_scored['latency_ms'].min() / summary_scored['latency_ms']
            else:
                summary_scored['norm_latency'] = 1.0
                
            if 'Num_MAE' in summary_scored.columns and summary_scored['Num_MAE'].max() > summary_scored['Num_MAE'].min():
                summary_scored['norm_mae'] = summary_scored['Num_MAE'].min() / summary_scored['Num_MAE']
            else:
                summary_scored['norm_mae'] = 1.0
            
            # Enhanced composite score with new metrics
            weights = {
                'norm_num_acc@tol': 0.25,  # Numeric accuracy
                'norm_em': 0.20,           # Exact match
                'norm_f1': 0.15,           # F1 score
                'norm_rouge_l': 0.10,      # ROUGE-L
                'norm_meteor': 0.10,       # METEOR
                'norm_bleu': 0.05,         # BLEU
                'norm_confidence': 0.08,    # Confidence
                'norm_latency': 0.05,      # Speed
                'norm_mae': 0.02           # Lower error
            }
            
            summary_scored['composite_score'] = sum(
                summary_scored.get(metric, 0) * weight 
                for metric, weight in weights.items()
            )
            
            # Sort by composite score
            ranking = summary_scored.sort_values('composite_score', ascending=False)
            
            print("\nRANKING (by enhanced composite score):")
            print("-" * 50)
            
            for i, (_, row) in enumerate(ranking.iterrows(), 1):
                print(f"{i}. 🥇 {row['system']} (Score: {row['composite_score']:.3f})")
                print(f"   • Numeric Accuracy: {row.get('Num_Acc@tol', 0):.1%}")
                print(f"   • Exact Match: {row.get('EM', 0):.1%}")
                print(f"   • F1 Score: {row.get('F1', 0):.3f}")
                if 'ROUGE_L' in row:
                    print(f"   • ROUGE-L: {row['ROUGE_L']:.3f}")
                if 'METEOR' in row:
                    print(f"   • METEOR: {row['METEOR']:.3f}")
                if 'BLEU' in row:
                    print(f"   • BLEU: {row['BLEU']:.3f}")
                print(f"   • Confidence: {row.get('confidence', 0):.3f}")
                print(f"   • Latency: {row.get('latency_ms', 0):.1f} ms")
                if 'Num_MAE' in row:
                    print(f"   • MAE: {row['Num_MAE']:.3f}")
                print()
            
            # Final recommendation
            best_system = ranking.iloc[0]['system']
            best_score = ranking.iloc[0]['composite_score']
            
            print("="*60)
            print(f"🏆 RECOMMENDED MODEL: {best_system.upper()}")
            print(f"📊 Enhanced Composite Score: {best_score:.3f}/1.000")
            print("="*60)
        
        # Enhanced recommendations for improvement
        print(f"\n" + "="*60)
        print("RECOMMENDATIONS FOR IMPROVEMENT")
        print("="*60)
        
        print("\n📈 To improve performance:")
        
        # Text quality analysis
        if 'ROUGE_L' in summary.columns and summary['ROUGE_L'].mean() < 0.3:
            print("• ROUGE-L scores are low - consider improving text generation coherence")
        if 'METEOR' in summary.columns and summary['METEOR'].mean() < 0.2:
            print("• METEOR scores are low - focus on semantic similarity in answers")
        if 'BLEU' in summary.columns and summary['BLEU'].mean() < 0.1:
            print("• BLEU scores are low - improve n-gram overlap with reference answers")
            
        if summary.get('Num_Acc@tol', pd.Series([0])).mean() < 0.8:
            print("• Numeric accuracy is low - consider improving number extraction")
            print("• Review document chunking strategy for numeric content")
            print("• Tune fusion weights for numeric questions")
            
        if summary.get('EM', pd.Series([0])).mean() < 0.3:
            print("• Exact match is very low - consider:")
            print("  - Better text generation models")
            print("  - Improved context selection")
            print("  - Answer post-processing")
            
        if identical_performance:
            print("• Models are too similar - consider:")
            print("  - Different fusion weights/parameters")
            print("  - Larger evaluation dataset")
            print("  - More diverse question types")
            print("  - Different chunking strategies per fusion type")
        
        print(f"\n📊 Dataset insights:")
        print(f"• Total questions evaluated: {len(results) // len(summary)}")
        print(f"• Numeric questions: {len(results[results['gold_num_norm'].notna()]) // len(summary)}")
        print(f"• Text questions: {len(results[results['gold_num_norm'].isna()]) // len(summary)}")
        print(f"• Average latency: {summary.get('latency_ms', pd.Series([0])).mean():.1f} ms")
        
        # Text quality insights
        if 'ROUGE_L' in summary.columns:
            print(f"• Average ROUGE-L: {summary['ROUGE_L'].mean():.3f}")
        if 'METEOR' in summary.columns:
            print(f"• Average METEOR: {summary['METEOR'].mean():.3f}")
        if 'BLEU' in summary.columns:
            print(f"• Average BLEU: {summary['BLEU'].mean():.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🔬 ENHANCED MULTIMODAL FUSION MODEL EVALUATION & SELECTION")
    print("="*70)
    print("This will evaluate Early, Late, and Hybrid fusion models")
    print("and recommend the best performing model based on comprehensive metrics:")
    print("• Traditional: EM, F1, Numeric Accuracy")
    print("• Enhanced: ROUGE-L, METEOR, BLEU")
    print("="*70)
    
    # Step 1: Check servers
    if not check_servers():
        print("\n❌ Not all servers are running. Please start them first.")
        print("\nTo start servers:")
        print("Terminal 1: cd s:/Multimodal\\ Project/Evalution/common && python model_server.py")
        print("Terminal 2: cd s:/Multimodal\\ Project/Evalution/early && python main_earlyfusion.py") 
        print("Terminal 3: cd s:/Multimodal\\ Project/Evalution/late && python main_latefusion.py")
        print("Terminal 4: cd s:/Multimodal\\ Project/Evalution/hybrid && python main_hybridfusion.py")
        return
    
    # Step 2: Check document
    doc_path = check_document()
    if not doc_path:
        print("\n❌ Document not found. Please ensure xyz.pdf is in the documents folder.")
        return
    
    # Step 3: Test single query
    test_single_query()
    
    # Step 4: Ask user to proceed
    proceed = input("\nDo you want to run the full enhanced evaluation? (y/n): ").lower().strip()
    if proceed != 'y':
        print("Evaluation cancelled.")
        return
    
    # Step 5: Run evaluation
    print(f"\n🚀 Starting comprehensive evaluation with enhanced metrics...")
    if run_evaluation():
        # Step 6: Analyze results and recommend best model
        print(f"\n📊 Analyzing results with ROUGE-L, METEOR, and BLEU metrics...")
        analyze_results()
    else:
        print("❌ Evaluation failed. Check the error messages above.")

if __name__ == "__main__":
    main()