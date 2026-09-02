import { useState, useEffect } from "react";
import { getCategories } from "./api";
import CategoryPanel from "./components/CategoryPanel";
import DocumentPanel from "./components/DocumentPanel";
import ChatPanel from "./components/ChatPanel";

export default function App() {
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState(null);

  async function loadCategories() {
    try {
      const data = await getCategories();
      setCategories(data);
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    loadCategories();
  }, []);

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden" }}>
      <CategoryPanel
        categories={categories}
        selected={selectedCategory}
        onSelect={setSelectedCategory}
        onRefresh={loadCategories}
      />
      <DocumentPanel category={selectedCategory} onDocumentsChange={() => {}} />
      <ChatPanel categories={categories} selectedCategory={selectedCategory} />
    </div>
  );
}
